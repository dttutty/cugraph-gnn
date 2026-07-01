# Build, Test, And Package

## Local Source Build

Build the full local stack:

```bash
./build.sh
```

Build an individual target:

```bash
./build.sh libwholegraph
./build.sh pylibwholegraph
./build.sh wholegraph-torch
./build.sh cugraph-pyg
```

Install Python packages in editable mode:

```bash
./build.sh --pydevelop pylibwholegraph wholegraph-torch cugraph-pyg
```

## Tests

Native tests:

```bash
./build.sh libwholegraph wholegraph-tests
./ci/run_ctests.sh
```

Python tests:

```bash
uv run --project python/pylibwholegraph pytest python/pylibwholegraph/tests
uv run --project python/wholegraph-torch pytest python/wholegraph-torch/tests
uv run --project python/cugraph-pyg pytest python/cugraph-pyg/tests
```

## CI Package Scripts

The CI scripts use this package split:

- `ci/build_cpp.sh`: builds conda `libwholegraph` and `libwholegraph-tests`.
- `ci/build_python.sh`: builds conda `pylibwholegraph`.
- `ci/build_python_wholegraph-torch.sh`: builds conda `wholegraph-torch`.
- `ci/build_python_noarch.sh`: builds conda `cugraph-pyg`.
- `ci/test_cpp.sh`: tests installed native test binaries.
- `ci/test_python.sh`: tests installed Python conda packages.

Wheel build and test scripts live next to the conda scripts and follow the same
dependency order.
