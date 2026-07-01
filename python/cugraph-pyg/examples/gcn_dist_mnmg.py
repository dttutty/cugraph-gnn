# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-node, multi-GPU Graph Convolutional Network (GCN) example.

This example demonstrates how to train a GCN for node property prediction
(classification) on a large OGB dataset across multiple nodes and GPUs using
cuGraph-PyG. It loads the entire dataset on rank 0, partitions it across all
ranks, and then trains the model in a distributed fashion.

WARNING: For large datasets, this approach may exceed a single worker's host
memory during the initial loading and partitioning phase. Consider pre-partitioning
the data if you encounter out-of-memory errors. Even the intermediate buffers
required during partitioning can be substantial.

Can be run with: torchrun --nproc-per-node=<num_gpus> gcn_dist_mnmg.py
"""

import argparse
import json
import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

import torch_geometric

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
from wholegraph_torch.initialize import (
    init as wg_init,
    finalize as wg_finalize,
)

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
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

    wg_init(global_rank, world_size, local_rank, torch.cuda.device_count())


def partition_data(dataset, split_idx, edge_path, feature_path, label_path, meta_path):
    data = dataset[0]

    # Split and save edge index
    os.makedirs(
        edge_path,
        exist_ok=True,
    )
    for r, e in enumerate(torch.tensor_split(data.edge_index, world_size, dim=1)):
        rank_path = os.path.join(edge_path, f"rank={r}.pt")
        torch.save(
            e.clone(),
            rank_path,
        )

    # Split and save features
    os.makedirs(
        feature_path,
        exist_ok=True,
    )

    for r, f in enumerate(torch.tensor_split(data.x, world_size)):
        rank_path = os.path.join(feature_path, f"rank={r}_x.pt")
        torch.save(
            f.clone(),
            rank_path,
        )
    for r, f in enumerate(torch.tensor_split(data.y, world_size)):
        rank_path = os.path.join(feature_path, f"rank={r}_y.pt")
        torch.save(
            f.clone(),
            rank_path,
        )

    # Split and save labels
    os.makedirs(
        label_path,
        exist_ok=True,
    )
    for d, i in split_idx.items():
        i_parts = torch.tensor_split(i, world_size)
        for r, i_part in enumerate(i_parts):
            rank_path = os.path.join(label_path, f"rank={r}")
            os.makedirs(rank_path, exist_ok=True)
            torch.save(i_part, os.path.join(rank_path, f"{d}.pt"))

    # Save metadata
    meta = {
        "num_classes": int(dataset.num_classes),
        "num_features": int(dataset.num_features),
        "num_nodes": int(data.num_nodes),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_partitioned_data(
    rank,
    edge_path,
    feature_path,
    label_path,
    meta_path,
    *,
    storage_backend,
    storage_location,
):
    from cugraph_pyg.data import GraphStore, FeatureStore
    from wholegraph_torch.storage_backend import resolve_storage_policy

    storage_policy = resolve_storage_policy(
        backend=storage_backend,
        location=storage_location,
    )
    if rank == 0:
        print(
            "WholeGraph storage policy: "
            f"backend={storage_policy.backend}, location={storage_policy.location}",
            flush=True,
        )
    graph_store = GraphStore(storage_policy=storage_policy)
    feature_store = FeatureStore(storage_policy=storage_policy)

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load labels
    split_idx = {}
    for split in ["train", "test", "valid"]:
        split_idx[split] = torch.load(
            os.path.join(label_path, f"rank={rank}", f"{split}.pt")
        )

    # Load features
    feature_store["node", "x", None] = torch.load(
        os.path.join(feature_path, f"rank={rank}_x.pt")
    )
    feature_store["node", "y", None] = torch.load(
        os.path.join(feature_path, f"rank={rank}_y.pt")
    )

    # Load edge index
    eix = torch.load(os.path.join(edge_path, f"rank={rank}.pt"))
    graph_store[
        ("node", "rel", "node"), "coo", False, (meta["num_nodes"], meta["num_nodes"])
    ] = eix

    return (feature_store, graph_store), split_idx, meta


def run_train(
    global_rank,
    data,
    split_idx,
    device,
    model,
    epochs,
    batch_size,
    fan_out,
    wall_clock_start,
    num_layers=3,
    seeds_per_call=-1,
    *,
    profile_train_steps=False,
    profile_every=1,
    max_train_batches=-1,
    max_eval_batches=-1,
    skip_eval=False,
    skip_test=False,
    torch_profile_dir=None,
    torch_profile_wait=1,
    torch_profile_warmup=1,
    torch_profile_active=3,
    torch_profile_repeat=1,
    torch_profile_record_shapes=False,
    torch_profile_memory=False,
    torch_profile_with_stack=False,
    torch_profile_with_flops=False,
):
    if os.getenv("CI", "false").lower() == "true" and seeds_per_call <= 0:
        warnings.warn("Detected CI environment; setting seeds_per_call to 20000")
        seeds_per_call = 20000

    enable_sampler_stage_profiling(profile_train_steps or bool(torch_profile_dir))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    from cugraph_pyg.loader import NeighborLoader

    ix_train = split_idx["train"].cuda()
    train_loader = NeighborLoader(
        data,
        input_nodes=ix_train,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    ix_test = split_idx["test"].cuda()
    test_loader = NeighborLoader(
        data,
        input_nodes=ix_test,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=min(seeds_per_call, 80000)
        if seeds_per_call > 0
        else 80000,
        **kwargs,
    )

    ix_valid = split_idx["valid"].cuda()
    valid_loader = NeighborLoader(
        data,
        input_nodes=ix_valid,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    dist.barrier()

    torch.cuda.synchronize()

    if global_rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time, "seconds")
        print("Beginning training...")
        if torch_profile_dir is not None:
            print(f"Torch profiler traces will be written under {torch_profile_dir}")

    total_train_time = 0
    total_val_time = 0
    for epoch in range(epochs):
        torch.cuda.synchronize()
        start = time.time()
        train_batches = 0
        train_iterator = iter(train_loader)
        i = 0
        while not limited_batch(max_train_batches, i):
            emit_step = (
                profile_train_steps and profile_every > 0 and i % profile_every == 0
            )
            torch_profile_this_batch = bool(
                torch_profile_dir
            ) and should_torch_profile_iteration(
                i,
                torch_profile_wait,
                torch_profile_warmup,
                torch_profile_active,
                torch_profile_repeat,
            )
            profile_this_batch = emit_step or torch_profile_this_batch

            with torch_profile_batch_context(
                torch_profile_this_batch,
                torch_profile_record_shapes,
                torch_profile_memory,
                torch_profile_with_stack,
                torch_profile_with_flops,
            ) as torch_profiler:
                marker_enabled = sample_nvtx_enabled("GCN", global_rank, i)
                if marker_enabled:
                    print(
                        f"GCN_SAMPLE_BEGIN rank={global_rank} "
                        f"epoch={epoch} iteration={i}",
                        flush=True,
                    )

                step_start = time.perf_counter()
                try:
                    with nvtx_range(
                        marker_enabled,
                        f"GCN_SAMPLE epoch={epoch} iteration={i}",
                    ):
                        batch, loader_total_s = timed_next(
                            train_iterator,
                            profile_this_batch,
                            "train/loader_total",
                        )
                except StopIteration:
                    break

                if marker_enabled:
                    print(
                        f"GCN_SAMPLE_END rank={global_rank} "
                        f"epoch={epoch} iteration={i}",
                        flush=True,
                    )

                timings = batch_sampler_stage_timings(batch, loader_total_s)

                if profile_this_batch:
                    sync_cuda()
                transfer_start = time.perf_counter()
                with record_function(torch_profile_this_batch, "train/transfer"):
                    batch = batch.to(device)
                if profile_this_batch:
                    sync_cuda()
                    timings["transfer"] = time.perf_counter() - transfer_start

                batch_size = batch.batch_size

                batch.y = batch.y.view(-1).to(torch.long)

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
                    out = model(batch.x, batch.edge_index)
                    loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
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
                    timings["total_iter"] = time.perf_counter() - step_start

            export_torch_profile_trace(
                torch_profiler,
                torch_profile_dir,
                global_rank,
                epoch,
                i,
            )

            if emit_step:
                emit_profile(
                    "GCN_PROFILE",
                    global_rank,
                    device,
                    "train",
                    epoch,
                    i,
                    timings,
                    batch,
                    extra={"loss": float(loss.detach().item())},
                    max_total_name="total_iter",
                )

            if global_rank == 0 and i % 10 == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + ", Iteration: "
                    + str(i)
                    + ", Loss: "
                    + str(loss)
                )
            i += 1
            train_batches += 1

        if global_rank == 0:
            end = time.time()
            total_train_time += end - start
            print(f"Epoch {epoch} train time: {end - start} s")
            if train_batches > 0:
                print(
                    "Average Training Iteration Time:",
                    (end - start) / train_batches,
                    "s/iter",
                )
            else:
                print("Average Training Iteration Time: no training batches")

        if not skip_eval:
            with torch.no_grad():
                total_correct = total_examples = 0
                torch.cuda.synchronize()
                start = time.time()
                for i, batch in enumerate(valid_loader):
                    if limited_batch(max_eval_batches, i):
                        break
                    batch = batch.to(device)
                    batch_size = batch.batch_size

                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)[:batch_size]

                    pred = out.argmax(dim=-1)
                    y = batch.y[:batch_size].view(-1).to(torch.long)

                    total_correct += int((pred == y).sum())
                    total_examples += y.size(0)

                acc_val = total_correct / total_examples if total_examples else 0.0
                if global_rank == 0:
                    end = time.time()
                    total_val_time += end - start
                    print(f"Epoch {epoch} val time: {end - start} s")
                    print(
                        f"Validation Accuracy: {acc_val * 100.0:.4f}%",
                    )

        torch.cuda.synchronize()

    if not skip_test:
        with torch.no_grad():
            total_correct = total_examples = 0
            for i, batch in enumerate(test_loader):
                if limited_batch(max_eval_batches, i):
                    break
                batch = batch.to(device)
                batch_size = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            acc_test = total_correct / total_examples if total_examples else 0.0
            if global_rank == 0:
                print(
                    f"Test Accuracy: {acc_test * 100.0:.4f}%",
                )

    if global_rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print(f"Train time: {total_train_time} s")
        print(f"Eval time: {total_val_time} s")
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")

    wg_finalize()

    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=30)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--skip_partition", action="store_true")
    parser.add_argument("--storage_backend", choices=["nccl", "vmm"], default=None)
    parser.add_argument("--storage_location", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument("--seeds_per_call", type=int, default=-1)
    parser.add_argument("--profile_train_steps", action="store_true")
    parser.add_argument("--profile_every", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--skip_eval", action="store_true")
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wall_clock_start = time.perf_counter()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)

        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            from pylibcugraph.comms import (
                cugraph_comms_create_unique_id,
            )

            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        dist.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]

        init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

        # Split the data
        edge_path = os.path.join(args.dataset_root, args.dataset + "_eix_part")
        feature_path = os.path.join(args.dataset_root, args.dataset + "_fea_part")
        label_path = os.path.join(args.dataset_root, args.dataset + "_label_part")
        meta_path = os.path.join(args.dataset_root, args.dataset + "_meta.json")

        # We partition the data to avoid loading it in every worker, which will
        # waste memory and can lead to an out of memory error.
        # cugraph_pyg.GraphStore and cugraph_pyg.FeatureStore are always
        # constructed from partitions of the edge index and features, respectively,
        # so this works well.
        if not args.skip_partition and global_rank == 0:
            # WARNING: The following code loads the entire dataset into rank 0's memory.
            # For large datasets, ensure you have sufficient RAM. The OGB datasets can
            # be several GB in size, and temporary buffers during partitioning may
            # require additional memory.
            with torch.serialization.safe_globals(
                [
                    torch_geometric.data.data.DataEdgeAttr,
                    torch_geometric.data.data.DataTensorAttr,
                    torch_geometric.data.storage.GlobalStorage,
                ]
            ):
                dataset = PygNodePropPredDataset(
                    name=args.dataset, root=args.dataset_root
                )
                split_idx = dataset.get_idx_split()

            # Partition and distribute the data across all ranks.
            partition_data(
                dataset,
                split_idx,
                meta_path=meta_path,
                label_path=label_path,
                feature_path=feature_path,
                edge_path=edge_path,
            )

        dist.barrier()
        from rmm.allocators.torch import rmm_torch_allocator

        with torch.cuda.use_mem_pool(
            torch.cuda.MemPool(rmm_torch_allocator.allocator())
        ):
            data, split_idx, meta = load_partitioned_data(
                rank=global_rank,
                edge_path=edge_path,
                feature_path=feature_path,
                label_path=label_path,
                meta_path=meta_path,
                storage_backend=args.storage_backend,
                storage_location=args.storage_location,
            )
            dist.barrier()

            model = torch_geometric.nn.models.GCN(
                meta["num_features"],
                args.hidden_channels,
                args.num_layers,
                meta["num_classes"],
            ).to(device)
            model = DistributedDataParallel(model, device_ids=[local_rank])

            run_train(
                global_rank,
                data,
                split_idx,
                device,
                model,
                args.epochs,
                args.batch_size,
                args.fan_out,
                wall_clock_start,
                args.num_layers,
                args.seeds_per_call,
                profile_train_steps=args.profile_train_steps,
                profile_every=args.profile_every,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                skip_eval=args.skip_eval,
                skip_test=args.skip_test,
                torch_profile_dir=args.torch_profile_dir,
                torch_profile_wait=args.torch_profile_wait,
                torch_profile_warmup=args.torch_profile_warmup,
                torch_profile_active=args.torch_profile_active,
                torch_profile_repeat=args.torch_profile_repeat,
                torch_profile_record_shapes=args.torch_profile_record_shapes,
                torch_profile_memory=args.torch_profile_memory,
                torch_profile_with_stack=args.torch_profile_with_stack,
                torch_profile_with_flops=args.torch_profile_with_flops,
            )
    else:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
