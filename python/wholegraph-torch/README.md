# wholegraph-torch

PyTorch adapters for WholeGraph distributed tensors, embeddings, graph
sampling helpers, and training utilities.

The low-level nanobind bindings live in `pylibwholegraph`; this package keeps
the PyTorch-facing layer separate from that binding boundary.
