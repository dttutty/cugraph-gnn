/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gather_func_impl.cuh"

namespace wholegraph_tensor_ops {

WHOLEGRAPH_DEFINE_GATHER_FUNC_IMPL(GatherFuncFloatingInt64,
                                   gather_floating_int64_temp_func,
                                   gather_floating_int64_func,
                                   int64_t,
                                   WHOLEGRAPH_DT_INT64,
                                   wholegraph_dtype_is_floating_number,
                                   ALLFLOAT,
                                   "gather CUDA LOGIC Error %s\n")

}  // namespace wholegraph_tensor_ops
