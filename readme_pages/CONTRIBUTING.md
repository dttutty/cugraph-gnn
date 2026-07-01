# Contributing

This repository contains the GNN-facing cuGraph package stack:

- `libwholegraph`
- `pylibwholegraph`
- `wholegraph-torch`
- `cugraph-pyg`

## Local Checks

Run style checks from the repository root:

```bash
pre-commit run --all-files
```

For Python-only iteration, prefer `uv run` from the repository root or from the
package directory being changed.

## Build Order

The local dependency order is:

```text
libwholegraph -> pylibwholegraph -> wholegraph-torch -> cugraph-pyg
```

The default build follows this order:

```bash
./build.sh
```

## Pull Requests

Changes should include focused tests for the package or native layer being
modified. Packaging changes should keep `dependencies.yaml`, package
`pyproject.toml` files, conda recipes, and CI scripts consistent.
