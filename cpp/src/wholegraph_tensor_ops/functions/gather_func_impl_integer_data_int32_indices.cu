/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gather_func_impl.cuh"

namespace wholegraph_tensor_ops {

WHOLEGRAPH_DEFINE_GATHER_FUNC_IMPL(GatherFuncIntegerInt32,
                                   gather_integer_int32_temp_func,
                                   gather_integer_int32_func,
                                   int32_t,
                                   WHOLEGRAPH_DT_INT,
                                   wholegraph_dtype_is_integer_number,
                                   ALLSINT,
                                   "gather LOGIC Error %s\n")

}  // namespace wholegraph_tensor_ops
