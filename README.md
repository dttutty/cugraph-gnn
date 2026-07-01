# cuGraph GNN

This repository owns the GNN-facing cuGraph stack.

It keeps WholeGraph storage, the WholeGraph PyTorch adapter, and the PyG
integration packages together. Native cuGraph graph algorithms and sampling
primitives live in the sibling `cugraph` repository and are consumed through
`pylibcugraph`.

## Packages

- `libwholegraph`: native WholeGraph runtime.
- `pylibwholegraph`: Python bindings for WholeGraph.
- `wholegraph-torch`: PyTorch adapter for WholeGraph storage and embeddings.
- `cugraph-pyg`: PyG graph store, feature store, loaders, samplers, and
  examples.

## Layout

- `cpp`: native WholeGraph sources, public headers, tests, and benchmarks.
- `python/libwholegraph`: wheel wrapper for the native WholeGraph runtime.
- `python/pylibwholegraph`: Python bindings and tests.
- `python/wholegraph-torch`: PyTorch adapter package.
- `python/cugraph-pyg`: PyG integration package, examples, and tests.
- `conda/recipes`: package recipes for the owned distributions.

## Dependency Direction

`cugraph-gnn` depends on the sibling `cugraph` repo for `pylibcugraph` and the
native sampling APIs. It owns the WholeGraph side of the stack locally:

```text
cugraph/libcugraph -> cugraph/pylibcugraph -> cugraph-gnn/cugraph-pyg
cugraph-gnn/libwholegraph -> cugraph-gnn/pylibwholegraph
                            -> cugraph-gnn/wholegraph-torch
                            -> cugraph-gnn/cugraph-pyg
```

## Build

The no-argument build follows the local GNN dependency order:

```bash
./build.sh
```

Equivalent explicit targets:

```bash
./build.sh libwholegraph pylibwholegraph wholegraph-torch cugraph-pyg
```

## Multi-GPU Development Notes

When running MG tests with a PyTorch wheel that bundles a different NCCL than
the RAPIDS/conda environment, preload the conda NCCL before importing PyTorch:

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libnccl.so.2
```

Always clean up cuGraph comms, WholeGraph, and `torch.distributed` in a
`finally` block for spawned MG tests. A failed assertion on one rank can
otherwise leave another rank blocked in a collective.

## Developer Docs

- [Contributing](readme_pages/CONTRIBUTING.md)
- [Build, test, and package](readme_pages/build_test_package.md)
- [WholeGraph C++ developer guide](cpp/docs/DEVELOPER_GUIDE.md)
