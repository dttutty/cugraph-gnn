/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholegraph/global_reference.h>

#ifdef __cplusplus
extern "C" {
#endif

wholegraph_gref_t wholegraph_create_continuous_global_reference(void* ptr)
{
  wholegraph_gref_t gref{};
  gref.pointer = ptr;
  return gref;
}

#ifdef __cplusplus
}
#endif
