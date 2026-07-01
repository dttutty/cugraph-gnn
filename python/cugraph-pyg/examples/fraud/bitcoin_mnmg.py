# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This example trains a GNN model on the EllipticBitcoin dataset
# using cuGraph-PyG.
# If an embedding directory is specified, the embeddings will be saved to a
# parquet file for downstream inspection.

import os
import argparse
import time

import torch

from torch_geometric.datasets import EllipticBitcoinDataset

from torch_geometric.nn.models import GraphSAGE, GCN, GAT

import torch.nn.functional as F

from cugraph_pyg.utils.profiling import (
    batch_sampler_stage_timings,
    emit_profile,
    enable_sampler_stage_profiling,
    export_torch_profile_trace,
    limited_batch,
    nvtx_range,
    record_function,
    sample_nvtx_enabled,
    should_torch_profile_iteration,
    sync_cuda,
    timed_next,
    torch_profile_batch_context,
)


def create_uid(global_rank, device):
    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        from pylibcugraph.comms import cugraph_comms_create_unique_id

        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]

    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)

    cugraph_id = cugraph_id[0]
    return cugraph_id


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=False,
        pool_allocator=False,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    torch.cuda.set_device(local_rank)

    from pylibcugraph.comms import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    # WholeGraph is already initialized.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./data/")
    parser.add_argument("--encoder", type=str, default="sage")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--embedding_dir", type=str, default=None, required=False)
    parser.add_argument("--profile_train_steps", action="store_true")
    parser.add_argument("--profile_every", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--max_embedding_batches", type=int, default=-1)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--torch_profile_dir", type=str, default=None)
    parser.add_argument("--torch_profile_wait", type=int, default=1)
    parser.add_argument("--torch_profile_warmup", type=int, default=1)
    parser.add_argument("--torch_profile_active", type=int, default=3)
    parser.add_argument("--torch_profile_repeat", type=int, default=1)
    parser.add_argument("--torch_profile_record_shapes", action="store_true")
    parser.add_argument("--torch_profile_memory", action="store_true")
    parser.add_argument("--torch_profile_with_stack", action="store_true")
    parser.add_argument("--torch_profile_with_flops", action="store_true")
    args = parser.parse_args()

    enable_sampler_stage_profiling(
        args.profile_train_steps or bool(args.torch_profile_dir)
    )

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    cugraph_id = create_uid(rank, device=torch.device(f"cuda:{local_rank}"))
    init_pytorch_worker(rank, local_rank, world_size, cugraph_id)

    dataset = EllipticBitcoinDataset(root=args.dataset_root)
    data = dataset[0]
    assert dataset.num_classes == 2

    data.x = data.x[:, :94]  # Remove pre-generated graph embeddings

    from cugraph_pyg.data import GraphStore, FeatureStore
    from wholegraph_torch import empty

    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Distribute data (will evenly distribute from rank 0;
    # all other ranks pass empties and receive their slice)
    graph_store[
        ("entity", "transaction", "entity"),
        "coo",
        False,
        (data.num_nodes, data.num_nodes),
    ] = data.edge_index if rank == 0 else empty(dim=2)
    feature_store["entity", "x", None] = data.x if rank == 0 else empty(dim=2)
    feature_store["entity", "y", None] = data.y if rank == 0 else empty(dim=1)
    torch.distributed.barrier()

    if args.encoder.lower() == "sage":
        encoder = GraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
            jk="last",
        )
    elif args.encoder.lower() == "gcn":
        encoder = GCN(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
            jk="last",
        )
    elif args.encoder.lower() == "gat":
        encoder = GAT(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
            jk="last",
        )
    else:
        raise ValueError(f"Invalid encoder: {args.encoder}")

    encoder = torch.nn.parallel.DistributedDataParallel(
        encoder.cuda(), device_ids=[local_rank]
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    ix_train = torch.tensor_split(
        torch.arange(data.num_nodes, device="cuda")[data.train_mask], world_size
    )[rank]

    ix_test = torch.tensor_split(
        torch.arange(data.num_nodes, device="cuda")[data.test_mask], world_size
    )[rank]

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_neighbors": [25, 10],
        "shuffle": True,
        "drop_last": True,
    }

    from cugraph_pyg.loader import NeighborLoader

    train_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_train,
        **loader_kwargs,
    )

    test_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_test,
        **loader_kwargs,
    )

    if rank == 0 and args.torch_profile_dir is not None:
        print(f"Torch profiler traces will be written under {args.torch_profile_dir}")

    for epoch in range(1, args.epochs + 1):
        train_iterator = iter(train_loader)
        it = 0
        while not limited_batch(args.max_train_batches, it):
            emit_step = (
                args.profile_train_steps
                and args.profile_every > 0
                and it % args.profile_every == 0
            )
            torch_profile_this_batch = bool(
                args.torch_profile_dir
            ) and should_torch_profile_iteration(
                it,
                args.torch_profile_wait,
                args.torch_profile_warmup,
                args.torch_profile_active,
                args.torch_profile_repeat,
            )
            profile_this_batch = emit_step or torch_profile_this_batch

            with torch_profile_batch_context(
                torch_profile_this_batch,
                args.torch_profile_record_shapes,
                args.torch_profile_memory,
                args.torch_profile_with_stack,
                args.torch_profile_with_flops,
            ) as torch_profiler:
                marker_enabled = sample_nvtx_enabled("BITCOIN", rank, it)
                if marker_enabled:
                    print(
                        f"BITCOIN_SAMPLE_BEGIN rank={rank} "
                        f"epoch={epoch} iteration={it}",
                        flush=True,
                    )

                step_start = time.perf_counter()
                try:
                    with nvtx_range(
                        marker_enabled,
                        f"BITCOIN_SAMPLE epoch={epoch} iteration={it}",
                    ):
                        batch, sample_s = timed_next(
                            train_iterator,
                            profile_this_batch,
                            "train/sample_and_feature_fetch",
                        )
                except StopIteration:
                    break

                if marker_enabled:
                    print(
                        f"BITCOIN_SAMPLE_END rank={rank} epoch={epoch} iteration={it}",
                        flush=True,
                    )

                timings = batch_sampler_stage_timings(batch, sample_s)

                if profile_this_batch:
                    sync_cuda()
                zero_grad_start = time.perf_counter()
                with record_function(torch_profile_this_batch, "train/zero_grad"):
                    optimizer.zero_grad()
                if profile_this_batch:
                    sync_cuda()
                    timings["zero_grad"] = time.perf_counter() - zero_grad_start

                if profile_this_batch:
                    sync_cuda()
                forward_start = time.perf_counter()
                with record_function(torch_profile_this_batch, "train/forward_loss"):
                    out = encoder(batch.x, batch.edge_index)

                    # only the initial batch is labeled
                    loss = F.cross_entropy(
                        out[: batch.batch_size], batch.y[: batch.batch_size]
                    )
                if profile_this_batch:
                    sync_cuda()
                    timings["forward_loss"] = time.perf_counter() - forward_start

                if profile_this_batch:
                    sync_cuda()
                backward_start = time.perf_counter()
                with record_function(torch_profile_this_batch, "train/backward"):
                    loss.backward()
                if profile_this_batch:
                    sync_cuda()
                    timings["backward"] = time.perf_counter() - backward_start

                if profile_this_batch:
                    sync_cuda()
                optimizer_start = time.perf_counter()
                with record_function(torch_profile_this_batch, "train/optimizer"):
                    optimizer.step()
                if profile_this_batch:
                    sync_cuda()
                    timings["optimizer"] = time.perf_counter() - optimizer_start
                    timings["train_step"] = time.perf_counter() - step_start

            export_torch_profile_trace(
                torch_profiler,
                args.torch_profile_dir,
                rank,
                epoch,
                it,
            )

            if emit_step:
                emit_profile(
                    "BITCOIN_PROFILE",
                    rank,
                    torch.device(f"cuda:{local_rank}"),
                    "train",
                    epoch,
                    it,
                    timings,
                    batch,
                    extra={"loss": float(loss.detach().item())},
                    max_total_name="train_step",
                )

            if rank == 0 and it % 10 == 0:
                print(f"Epoch {epoch} iter {it} loss: {loss.item()}")
            it += 1

    if not args.skip_test:
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            for it, batch in enumerate(test_loader):
                if limited_batch(args.max_eval_batches, it):
                    break
                out = encoder(batch.x, batch.edge_index)

                loss = F.cross_entropy(
                    out[: batch.batch_size], batch.y[: batch.batch_size]
                )
                total_loss += loss.item() * batch.batch_size
                total_examples += batch.batch_size
                total_correct += (
                    (
                        out[: batch.batch_size].argmax(dim=-1)
                        == batch.y[: batch.batch_size]
                    )
                    .sum()
                    .item()
                )
            loss_value = total_loss / total_examples if total_examples else 0.0
            acc_value = total_correct / total_examples if total_examples else 0.0
            print(f"rank={rank} Test loss: {loss_value} acc: {acc_value}")

    torch.distributed.barrier()

    if args.embedding_dir is not None:
        inf_loader = NeighborLoader(
            (feature_store, graph_store),
            input_nodes=torch.tensor_split(
                torch.arange(data.num_nodes, device="cuda"), world_size
            )[rank],
            num_neighbors=[-1],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        feature_store["entity", "emb", None] = (
            torch.zeros(
                (data.num_nodes, args.hidden_channels),
                dtype=torch.float32,
                device="cuda",
            )
            if rank == 0
            else empty(dim=2)
        )

        feature_store["entity", "z", None] = (
            torch.zeros((data.num_nodes,), dtype=torch.float32, device="cuda")
            if rank == 0
            else empty(dim=1)
        )

        with torch.no_grad():
            for it, batch in enumerate(inf_loader):
                if limited_batch(args.max_embedding_batches, it):
                    break
                x = batch.x
                edge_index = batch.edge_index
                for layer, (conv, norm) in enumerate(
                    zip(encoder.module.convs, encoder.module.norms)
                ):
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = encoder.module.act(x)

                z = encoder.module.lin(x)[: batch.batch_size].softmax(dim=-1)[:, 0]

                x = x[: batch.batch_size]
                feature_store["entity", "emb", None][batch.n_id[: batch.batch_size]] = x
                feature_store["entity", "z", None][batch.n_id[: batch.batch_size]] = z

        import cudf

        df = cudf.DataFrame(
            feature_store["entity", "emb", None].get_local_tensor(),
            index=None,
            columns=[f"emb_{i}" for i in range(args.hidden_channels)],
        )
        df["y"] = (feature_store["entity", "y", None]).get_local_tensor()
        df["z"] = (feature_store["entity", "z", None]).get_local_tensor()

        os.makedirs(args.embedding_dir, exist_ok=True)
        df.to_parquet(
            f"{args.embedding_dir}/emb_{args.encoder}_{args.hidden_channels}"
            f"_{args.batch_size}_{args.lr}_{args.epochs}_{rank}.parquet"
        )

    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()
    torch.distributed.destroy_process_group()
