# WholeGraph Native Runtime

WholeGraph provides the native storage, communication, tensor, embedding, and
sampling primitives used by the Python GNN-facing packages in this repository.

The installed public C++ headers live under `include/wholegraph`. Internal
implementation files live under `src/wholegraph`, `src/wholegraph_tensor_ops`,
`src/wholegraph_ops`, and `src/graph_ops`.
