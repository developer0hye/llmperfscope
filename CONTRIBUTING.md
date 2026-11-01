# Contributing

Issues and pull requests are welcome. Reports of a wrong number or a misleading verdict are especially useful: please attach the run, copied with `llmperfscope anonymize RUN --out DIR`.

## Setting up

```sh
git clone https://github.com/developer0hye/llmperfscope && cd llmperfscope
pip install -e ".[figures]"
cd tests && python3 -m unittest
```

The tests need Linux but no GPU: real processes with known behaviour stand in for an engine, and scripted GPUs for hardware. On other systems `test_samples` and `test_docs` still run.

## What a change should do

- **An instrument has to be right.** A new number needs a test where the true answer is known, because the process was made to behave that way. Then the report has to say what happened.
- **Keep the measuring side free of dependencies.** `pip install llmperfscope` installs nothing else. Only `figures` needs matplotlib, and it loads only when drawing.
- **Do not disturb the engine.** Anything slow goes on its own thread and at a low rate. If a change could cost the engine speed, show it does not: alternate fresh processes with and without it, on a real engine. See [docs/validation.md](docs/validation.md).
- **State the evidence.** A finding carries the numbers it rests on, and says so when it cannot tell. It stays silent on short runs and shared machines unless the evidence survives them.
- **Keep the docs in step.** A new option or finding goes in `docs/`. `tests/test_docs.py` checks that every option and relative link in the docs exists.

## Adding an engine

See [docs/adapters.md](docs/adapters.md). Often a preset is enough. Please add a real sample of the engine's output to `tests/test_parsers.py`.

## Figures

The series colors are validated for color vision deficiency in their fixed order, so assign them in sequence and never cycle them. Render at `--width single` and `--width double` and look at the result, since tests cannot see an overlap. Regenerate the gallery with `docs/gallery/make.sh`.

## Sample runs

`examples/sample-runs` holds real runs, copied with `llmperfscope anonymize`. Check a new one for names before adding it.
