"""Keeps the GPU busy for a known time through Metal, with nothing but the system's own libraries. A test double with a known answer."""

import ctypes as C
import sys
import time

objc = C.CDLL("/usr/lib/libobjc.dylib")
metal = C.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
C.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
objc.objc_getClass.restype = C.c_void_p
objc.objc_getClass.argtypes = [C.c_char_p]
objc.sel_registerName.restype = C.c_void_p
objc.sel_registerName.argtypes = [C.c_char_p]
objc.objc_autoreleasePoolPush.restype = C.c_void_p
objc.objc_autoreleasePoolPop.argtypes = [C.c_void_p]
metal.MTLCreateSystemDefaultDevice.restype = C.c_void_p

SHADER = b"""
#include <metal_stdlib>
using namespace metal;
kernel void burn(device float *out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    float x = float(id);
    for (int i = 0; i < 4000; i++) x = sin(x) + cos(x + 1.0);
    out[id % 1024] = x;
}
"""


class Size(C.Structure):
    _fields_ = [("width", C.c_size_t), ("height", C.c_size_t), ("depth", C.c_size_t)]


def send(target, selector, *args, restype=C.c_void_p, argtypes=()):
    function = C.CFUNCTYPE(restype, C.c_void_p, C.c_void_p, *argtypes)(("objc_msgSend", objc))
    return function(target, objc.sel_registerName(selector), *args)


def string(text):
    return send(objc.objc_getClass(b"NSString"), b"stringWithUTF8String:", text, argtypes=[C.c_char_p])


def main(seconds):
    pool = objc.objc_autoreleasePoolPush()
    device = metal.MTLCreateSystemDefaultDevice()
    if not device:
        return 2
    library = send(device, b"newLibraryWithSource:options:error:", string(SHADER), None, None, argtypes=[C.c_void_p, C.c_void_p, C.c_void_p])
    function = send(library, b"newFunctionWithName:", string(b"burn"), argtypes=[C.c_void_p])
    pipeline = send(device, b"newComputePipelineStateWithFunction:error:", function, None, argtypes=[C.c_void_p, C.c_void_p])
    queue = send(device, b"newCommandQueue")
    buffer = send(device, b"newBufferWithLength:options:", 4096, 0, argtypes=[C.c_size_t, C.c_size_t])
    objc.objc_autoreleasePoolPop(pool)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pool = objc.objc_autoreleasePoolPush()
        commands = send(queue, b"commandBuffer")
        encoder = send(commands, b"computeCommandEncoder")
        send(encoder, b"setComputePipelineState:", pipeline, restype=None, argtypes=[C.c_void_p])
        send(encoder, b"setBuffer:offset:atIndex:", buffer, 0, 0, restype=None, argtypes=[C.c_void_p, C.c_size_t, C.c_size_t])
        send(encoder, b"dispatchThreads:threadsPerThreadgroup:", Size(1 << 16, 1, 1), Size(64, 1, 1), restype=None, argtypes=[Size, Size])
        send(encoder, b"endEncoding", restype=None)
        send(commands, b"commit", restype=None)
        send(commands, b"waitUntilCompleted", restype=None)
        objc.objc_autoreleasePoolPop(pool)
    return 0


if __name__ == "__main__":
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 3.0))
