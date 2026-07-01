# WholeGraph C++ Developer Guide

This directory owns the native WholeGraph runtime used by the Python packages in
this repository.

## Source Layout

- `include/wholegraph`: installed public headers.
- `src/wholegraph`: runtime, communicator, tensor, embedding, and memory handle
  implementation.
- `src/wholegraph_tensor_ops`: gather, scatter, embedding, and tensor operation
  kernels.
- `src/wholegraph_ops`: graph sampling kernels that operate on WholeGraph-backed
  tensors.
- `src/graph_ops`: smaller graph utility operations shared by tests and higher
  level APIs.
- `tests`: C++ and CUDA tests installed by the `libwholegraph-tests` conda
  package.
- `bench`: native benchmarks.

The old `wholememory` and `wholememory_ops` names are intentionally replaced by
`wholegraph` and `wholegraph_tensor_ops`. New code should not add files under
the old names.

## Build And Test

Configure and build the native runtime from the repository root:

```bash
./build.sh libwholegraph
```

Build native tests:

```bash
./build.sh libwholegraph wholegraph-tests
```

Run installed or locally built tests:

```bash
./ci/run_ctests.sh
```

Use `--allgpuarch` for release-style builds and the default native GPU
architecture for local iteration.

## Style

- Keep public ABI declarations in `include/wholegraph`.
- Keep CUDA device helpers in `.cuh` only when they are included by CUDA
  translation units.
- Prefer stream-ordered RAII wrappers and existing RMM/RAFT types over raw CUDA
  allocation.
- Keep C++ and CUDA formatting compatible with `cpp/.clang-format`.
- Add or update C++ tests when changing native behavior.

## Documentation

`cpp/Doxyfile` generates XML documentation for native headers and selected
developer docs. The generated output is not checked in.
