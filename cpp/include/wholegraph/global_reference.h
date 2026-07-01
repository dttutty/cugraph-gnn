/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Global reference of a WholeGraph object
 *
 * A global reference is for mapped WholeGraph. The global reference is used to directly access
 * memory visible to the current rank.
 */
struct wholegraph_gref_t {
  void* pointer; /*!< pointer to mapped data */
};

/**
 * @brief Create global reference for continuous memory
 * @param ptr : pointer to the memory
 * @return : wholegraph_gref_t
 */
wholegraph_gref_t wholegraph_create_continuous_global_reference(void* ptr);

#ifdef __cplusplus
}
#endif
