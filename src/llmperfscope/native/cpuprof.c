// In-process CPU sampling profiler, loaded with LD_PRELOAD. It needs no privileges: perf_event_paranoid does not apply to POSIX
// timers, so this works where `perf record` is refused.
//
// Each thread gets its own CPU-time timer. A single process-wide timer (setitimer) would not do: the kernel delivers its signal to
// the main thread whenever that thread is running, so work done in parallel would look like work done by the main thread alone,
// and "this function runs on one thread" is exactly the kind of thing this tool is used to find out.
//
// A signal only arrives when a system call returns. While a thread sits in one long call the timer keeps expiring and the kernel
// merges the expirations into one signal, with their number in si_overrun. The sample records that number, so it counts for the
// whole time: without it, a munmap of 12 s looks like one sample.
//
// Each sample: time, thread id and name, how many periods it stands for, interrupted PC, and the stack from the unwinder. Samples go
// to a preallocated buffer and reach disk at exit, so the run does no file I/O for profiling. llmperfscope reads the files and
// symbolizes them.
//
// It interposes pthread_create, to start the timer of each new thread, and nothing else.
//
// Environment:
//   LLMPERFSCOPE_CPUPROF_OUT  path prefix; files are <prefix>.<pid>.{bin,maps}
//   LLMPERFSCOPE_CPUPROF_HZ   samples per CPU second per thread, default 200
//
// Build: cc -O2 -shared -fPIC -pthread -o libllmperfscope_cpu.so cpuprof.c -ldl -lrt

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#define MAX_FRAMES 30
#define BUFFER_BYTES (512ull << 20)  // virtual; only touched pages count

typedef struct {
    uint64_t t_ns;
    uint32_t tid;
    uint32_t depth;
    uint32_t weight;
    uint32_t reserved;
    char comm[16];
    uint64_t frames[MAX_FRAMES];
} sample_t;

static sample_t *g_samples;
static uint64_t g_capacity;
static atomic_uint_fast64_t g_next;
static atomic_uint_fast64_t g_dropped;
static atomic_int g_armed;
static int g_hz = 200;
static char g_prefix[PATH_MAX];
static pid_t g_pid;
static pthread_key_t g_timer_key;

static __thread timer_t t_timer;
static __thread int t_has_timer;

static void on_sigprof(int signum, siginfo_t *info, void *context) {
    (void) signum;
    if (!atomic_load_explicit(&g_armed, memory_order_relaxed)) {
        return;
    }
    int saved_errno = errno;
    uint_fast64_t i = atomic_fetch_add_explicit(&g_next, 1, memory_order_relaxed);
    if (i >= g_capacity) {
        atomic_fetch_add_explicit(&g_dropped, 1, memory_order_relaxed);
        errno = saved_errno;
        return;
    }
    sample_t *s = &g_samples[i];
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    s->t_ns = (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
    s->tid = (uint32_t) syscall(SYS_gettid);
    s->weight = 1 + (info->si_code == SI_TIMER && info->si_overrun > 0 ? (uint32_t) info->si_overrun : 0);
    prctl(PR_GET_NAME, s->comm, 0, 0, 0);
    void *stack[MAX_FRAMES + 2];
    int n = backtrace(stack, MAX_FRAMES + 2);
    uint32_t depth = 0;
#if defined(__x86_64__)
    s->frames[depth++] = (uint64_t) ((ucontext_t *) context)->uc_mcontext.gregs[REG_RIP];
#else
    (void) context;
#endif
    // stack[0] is this handler and stack[1] the signal trampoline; stack[2] repeats the interrupted PC.
    for (int k = 3; k < n && depth < MAX_FRAMES; k++) {
        s->frames[depth++] = (uint64_t) stack[k];
    }
    s->depth = depth;
    errno = saved_errno;
}

static void arm_this_thread(void) {
    if (t_has_timer || !atomic_load(&g_armed)) {
        return;
    }
    struct sigevent event;
    memset(&event, 0, sizeof(event));
    event.sigev_notify = SIGEV_THREAD_ID;
    event.sigev_signo = SIGPROF;
    event._sigev_un._tid = (pid_t) syscall(SYS_gettid);
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &event, &t_timer) != 0) {
        return;
    }
    struct itimerspec spec;
    spec.it_interval.tv_sec = 0;
    spec.it_interval.tv_nsec = 1000000000L / g_hz;
    spec.it_value = spec.it_interval;
    timer_settime(t_timer, 0, &spec, NULL);
    t_has_timer = 1;
    pthread_setspecific(g_timer_key, (void *) 1);
}

// A timer does not go away with its thread on its own.
static void on_thread_exit(void *unused) {
    (void) unused;
    if (t_has_timer) {
        timer_delete(t_timer);
        t_has_timer = 0;
    }
}

typedef struct {
    void *(*start)(void *);
    void *arg;
} thunk_t;

static void *thunk_main(void *raw) {
    thunk_t thunk = *(thunk_t *) raw;
    free(raw);
    arm_this_thread();
    return thunk.start(thunk.arg);
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start)(void *), void *arg) {
    static int (*real)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    if (!real) {
        real = dlsym(RTLD_NEXT, "pthread_create");
    }
    if (!g_samples) {
        return real(thread, attr, start, arg);
    }
    thunk_t *thunk = malloc(sizeof(thunk_t));
    if (!thunk) {
        return real(thread, attr, start, arg);
    }
    thunk->start = start;
    thunk->arg = arg;
    int result = real(thread, attr, thunk_main, thunk);
    if (result != 0) {
        free(thunk);
    }
    return result;
}

static void copy_file(const char *from, const char *to) {
    int in = open(from, O_RDONLY);
    if (in < 0) {
        return;
    }
    int out = open(to, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out >= 0) {
        char buffer[65536];
        ssize_t n;
        while ((n = read(in, buffer, sizeof(buffer))) > 0) {
            if (write(out, buffer, (size_t) n) != n) {
                break;
            }
        }
        close(out);
    }
    close(in);
}

static void dump(void) {
    if (!g_samples || g_pid != getpid()) {
        return;
    }
    uint_fast64_t count = atomic_load(&g_next);
    if (count > g_capacity) {
        count = g_capacity;
    }
    if (count == 0) {
        return;
    }
    char path[PATH_MAX + 32];
    snprintf(path, sizeof(path), "%s.%d.maps", g_prefix, (int) g_pid);
    copy_file("/proc/self/maps", path);

    snprintf(path, sizeof(path), "%s.%d.bin", g_prefix, (int) g_pid);
    FILE *f = fopen(path, "wb");
    if (f) {
        uint64_t header[5] = {0x4c5053434f504532ull, (uint64_t) count, (uint64_t) atomic_load(&g_dropped), MAX_FRAMES, (uint64_t) g_hz};
        fwrite(header, sizeof(header), 1, f);
        fwrite(g_samples, sizeof(sample_t), count, f);
        fclose(f);
    }
}

// Wrappers exec the real engine right away; they are not worth a timer. POSIX timers do not survive exec, so nothing else is needed.
static int skip_this_process(void) {
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) {
        return 1;
    }
    exe[n] = 0;
    const char *base = strrchr(exe, '/');
    base = base ? base + 1 : exe;
    static const char *const skip[] = {"taskset", "numactl", "env", "timeout", "nice", "chrt", "flock", "sh", "bash", "dash",
                                       "nsys", "nvidia-smi", "git", "stdbuf", "setsid", NULL};
    for (int i = 0; skip[i]; i++) {
        if (strcmp(base, skip[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static void child_after_fork(void) {
    // Timers do not survive fork. A forked worker gets its own samples and its own files.
    g_pid = getpid();
    atomic_store(&g_next, 0);
    atomic_store(&g_dropped, 0);
    t_has_timer = 0;
    arm_this_thread();
}

__attribute__((constructor)) static void start(void) {
    const char *prefix = getenv("LLMPERFSCOPE_CPUPROF_OUT");
    if (!prefix || !*prefix || skip_this_process()) {
        return;
    }
    snprintf(g_prefix, sizeof(g_prefix), "%s", prefix);
    const char *hz = getenv("LLMPERFSCOPE_CPUPROF_HZ");
    if (hz && atoi(hz) > 0) {
        g_hz = atoi(hz);
    }
    void *memory = mmap(NULL, BUFFER_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (memory == MAP_FAILED) {
        return;
    }
    // The first backtrace() may allocate and load libgcc, which a signal handler must not do.
    void *warm[4];
    backtrace(warm, 4);
    if (pthread_key_create(&g_timer_key, on_thread_exit) != 0) {
        return;
    }
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = on_sigprof;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    sigaction(SIGPROF, &action, NULL);
    g_capacity = BUFFER_BYTES / sizeof(sample_t);
    g_pid = getpid();
    g_samples = (sample_t *) memory;
    atomic_store(&g_armed, 1);
    pthread_atfork(NULL, NULL, child_after_fork);
    arm_this_thread();
}

__attribute__((destructor)) static void stop(void) {
    if (g_samples) {
        // Timers may still fire; the handler stays installed and ignores them.
        atomic_store(&g_armed, 0);
        dump();
    }
}
