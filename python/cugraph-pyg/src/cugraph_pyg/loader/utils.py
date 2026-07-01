# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph_pyg.utils.imports import import_optional

torch = import_optional("torch")


def generate_seed():
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        seed = torch.randint(
            0, 2**63 - world_size, (1,), dtype=torch.int64, device="cuda"
        )
    else:
        seed = torch.tensor([0], dtype=torch.int64, device="cuda")
    torch.distributed.broadcast(seed, src=0)
    seed = seed.item() + rank
    return seed


def validate_input_batch_size(count, batch_size, drop_last, *, input_kind, input_name):
    if count < batch_size and drop_last:
        raise ValueError(
            f"The number of input {input_kind} is less than the batch size"
            " and drop_last is True. This will result in all batches"
            " being dropped. Either set drop_last to False or increase"
            f" the number of {input_kind} in {input_name}."
        )


def make_input_id(count, input_id):
    if input_id is not None:
        return input_id
    return torch.arange(count, dtype=torch.int64, device="cuda")


def get_input_permutation(count, shuffle, batch_size, drop_last):
    if shuffle:
        perm = torch.randperm(count)
    else:
        perm = torch.arange(count)

    if drop_last:
        dropped_count = perm.numel() % batch_size
        if dropped_count > 0:
            perm = perm[:-dropped_count]

    return perm


def index_optional(value, index):
    if value is None:
        return None
    return value[index]
