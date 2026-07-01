# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

KARATE_NUM_NODES = 34

_KARATE_SRC = [
    *range(KARATE_NUM_NODES),
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
]

_KARATE_DST = [
    *range(1, KARATE_NUM_NODES),
    0,
    5,
    12,
    8,
    15,
    20,
    25,
    30,
    1,
    3,
    6,
    9,
    11,
    13,
    17,
    19,
    23,
    27,
    31,
]


def karate_edgelist(torch, device="cuda"):
    return (
        torch.tensor(_KARATE_SRC, device=device, dtype=torch.int64),
        torch.tensor(_KARATE_DST, device=device, dtype=torch.int64),
    )
