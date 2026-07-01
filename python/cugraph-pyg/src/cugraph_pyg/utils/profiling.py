# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from contextlib import contextmanager, nullcontext

from cugraph_pyg.utils.imports import MissingModule, import_optional

torch = import_optional("torch")
dist = import_optional("torch.distributed")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def sample_nvtx_enabled(prefix: str, global_rank: int, iteration: int) -> bool:
    if not (
        env_flag(f"{prefix}_MARK_SAMPLE") or env_flag(f"{prefix}_MARK_FIRST_SAMPLE")
    ):
        return False

    all_ranks = env_flag(f"{prefix}_MARK_SAMPLE_ALL_RANKS")
    marker_rank = env_int(f"{prefix}_MARK_SAMPLE_RANK", 0)
    if not all_ranks and global_rank != marker_rank:
        return False

    limit = env_int(f"{prefix}_MARK_SAMPLE_LIMIT", 1)
    return limit < 0 or iteration < limit


def sync_cuda() -> None:
    if not isinstance(torch, MissingModule) and torch.cuda.is_available():
        torch.cuda.synchronize()


def record_function(enabled: bool, name: str):
    if not enabled or isinstance(torch, MissingModule):
        return nullcontext()

    profiler = getattr(torch, "profiler", None)
    record = getattr(profiler, "record_function", None)
    if record is None:
        return nullcontext()
    return record(name)


@contextmanager
def nvtx_range(enabled: bool, name: str):
    pushed = False
    if enabled and not isinstance(torch, MissingModule) and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        pushed = True
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def enable_sampler_stage_profiling(enabled: bool, prefix: str = "train") -> None:
    if enabled:
        os.environ.setdefault("CUGRAPH_PYG_PROFILE_SAMPLER_STAGES", "1")
        os.environ.setdefault("CUGRAPH_PYG_PROFILE_STAGE_PREFIX", prefix)


def limited_batch(max_batches: int, iteration: int) -> bool:
    return max_batches >= 0 and iteration >= max_batches


def rank_max_float(value: float, device) -> float:
    if (
        isinstance(dist, MissingModule)
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return float(value)

    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def rank_with_max_total(device, timings: dict[str, float], total_name: str):
    if total_name not in timings:
        return None

    names = list(timings)
    if (
        isinstance(dist, MissingModule)
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return 0, names, [float(timings[name]) for name in names]

    local = torch.tensor(
        [float(timings[name]) for name in names],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    stacked = torch.stack(gathered)
    total_index = names.index(total_name)
    rank = int(torch.argmax(stacked[:, total_index]).item())
    return rank, names, [float(value) for value in stacked[rank].tolist()]


def batch_profile_stats(batch) -> dict[str, int | float | None]:
    edge_index = getattr(batch, "edge_index", None)
    x = getattr(batch, "x", None)
    num_nodes = getattr(batch, "num_nodes", None)
    batch_size = getattr(batch, "batch_size", None)

    if num_nodes is None and x is not None:
        num_nodes = x.size(0)

    stats = {
        "batch_size": int(batch_size) if batch_size is not None else None,
        "num_nodes": int(num_nodes) if num_nodes is not None else None,
        "num_edges": int(edge_index.size(1)) if edge_index is not None else None,
    }
    feature_attrs = getattr(batch, "_cugraph_feature_gather_num_attrs", None)
    if feature_attrs is not None:
        stats["feature_gather_num_attrs"] = int(feature_attrs)
    return stats


def batch_sampler_stage_timings(batch, loader_total_s: float):
    timings = {"loader_total": float(loader_total_s)}
    for attr_name, timing_name in [
        ("_cugraph_sampler_core_s", "sampler_core"),
        ("_cugraph_batch_materialize_s", "batch_materialize"),
        ("_cugraph_feature_gather_s", "feature_gather"),
    ]:
        value = getattr(batch, attr_name, None)
        if value is not None:
            timings[timing_name] = float(value)

    accounted = 0.0
    for timing_name in ["sampler_core", "batch_materialize", "feature_gather"]:
        accounted += timings.get(timing_name, 0.0)
    if accounted > 0.0:
        timings["loader_unaccounted"] = max(0.0, timings["loader_total"] - accounted)

    return timings


def emit_profile(
    tag: str,
    global_rank: int,
    device,
    phase: str,
    epoch: int,
    iteration: int,
    timings: dict[str, float],
    batch,
    extra: dict | None = None,
    max_total_name: str | None = None,
) -> None:
    max_rank_timings = {
        f"{name}_s_max_rank": round(rank_max_float(value, device), 6)
        for name, value in timings.items()
    }

    total_rank_timings = {}
    if max_total_name is not None:
        selected = rank_with_max_total(device, timings, max_total_name)
        if selected is not None:
            max_rank, names, values = selected
            total_rank_timings[f"{max_total_name}_max_rank"] = int(max_rank)
            for i, name in enumerate(names):
                total_rank_timings[f"{name}_on_max_{max_total_name}_rank_s"] = round(
                    values[i], 6
                )

    if global_rank != 0:
        return

    payload = {
        "phase": phase,
        "epoch": int(epoch),
        "iteration": int(iteration),
    }
    payload.update(batch_profile_stats(batch))
    payload.update(max_rank_timings)
    payload.update(total_rank_timings)
    if extra:
        payload.update(extra)

    print(f"{tag} " + json.dumps(payload, sort_keys=True), flush=True)


def should_torch_profile_iteration(
    iteration: int,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
) -> bool:
    cycle = wait + warmup + active
    if active <= 0 or cycle <= 0:
        return False
    cycle_index = iteration // cycle
    if repeat > 0 and cycle_index >= repeat:
        return False
    return wait + warmup <= iteration % cycle < cycle


def torch_profile_batch_context(
    enabled: bool,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    with_flops: bool,
):
    if not enabled:
        return nullcontext()

    if isinstance(torch, MissingModule):
        return nullcontext()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    )


def export_torch_profile_trace(
    profiler,
    profile_dir: str | None,
    global_rank: int,
    epoch: int,
    iteration: int,
) -> None:
    if profiler is None or profile_dir is None:
        return

    rank_dir = os.path.join(profile_dir, f"rank{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)
    trace_path = os.path.join(
        rank_dir,
        f"rank{global_rank}_epoch{epoch}_iter{iteration}.trace.json",
    )
    profiler.export_chrome_trace(trace_path)
    if global_rank == 0:
        print(f"Torch profiler trace written: {trace_path}", flush=True)


def timed_next(iterator, profile_enabled: bool, record_name: str):
    if profile_enabled:
        sync_cuda()
    start = time.perf_counter()
    with record_function(profile_enabled, record_name):
        batch = next(iterator)
    if profile_enabled:
        sync_cuda()
    return batch, time.perf_counter() - start
