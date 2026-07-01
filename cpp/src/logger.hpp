/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdarg>

#include <iostream>
#include <string>

#include <cassert>
#include <raft/core/error.hpp>

#include "error.hpp"
#include <wholegraph/wholegraph.h>

namespace wholegraph {

LogLevel& get_log_level();

void set_log_level(LogLevel lev);

bool will_log_for(LogLevel lev);

/**
 * @defgroup CStringFormat Expand a C-style format string
 *
 * @brief Expands C-style formatted string into std::string
 *
 * @param[in] fmt format string
 * @param[in] vl  respective values for each of format modifiers in the string
 *
 * @return the expanded `std::string`
 *
 * @{
 */
inline std::string format(const char* fmt, va_list& vl)
{
  va_list vl_copy;
  va_copy(vl_copy, vl);
  int length = std::vsnprintf(nullptr, 0, fmt, vl_copy);
  assert(length >= 0);
  std::vector<char> buf(length + 1);
  (void)std::vsnprintf(buf.data(), length + 1, fmt, vl);
  return std::string(buf.data());
}

inline std::string format(const char* fmt, ...)
{
  va_list vl;
  va_start(vl, fmt);
  std::string str = wholegraph::format(fmt, vl);
  va_end(vl);
  return str;
}
/** @} */

#define WHOLEGRAPH_LOG(lev, fmt, ...)                                                 \
  do {                                                                                 \
    if (wholegraph::will_log_for(lev))                                                \
      std::cout << wholegraph::format(fmt, ##__VA_ARGS__) << std::endl << std::flush; \
  } while (0)

#define WHOLEGRAPH_FATAL(fmt, ...)                                                    \
  do {                                                                                 \
    std::string fatal_msg{};                                                           \
    SET_WHOLEGRAPH_ERROR_MSG(fatal_msg, "WholeGraph FATAL at ", fmt, ##__VA_ARGS__); \
    throw wholegraph::logic_error(fatal_msg);                                         \
  } while (0)

#define WHOLEGRAPH_ERROR(fmt, ...) WHOLEGRAPH_LOG(LEVEL_ERROR, fmt, ##__VA_ARGS__)
#define WHOLEGRAPH_WARN(fmt, ...)  WHOLEGRAPH_LOG(LEVEL_WARN, fmt, ##__VA_ARGS__)
#define WHOLEGRAPH_INFO(fmt, ...)  WHOLEGRAPH_LOG(LEVEL_INFO, fmt, ##__VA_ARGS__)
#define WHOLEGRAPH_DEBUG(fmt, ...) WHOLEGRAPH_LOG(LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define WHOLEGRAPH_TRACE(fmt, ...) WHOLEGRAPH_LOG(LEVEL_TRACE, fmt, ##__VA_ARGS__)

}  // namespace wholegraph
