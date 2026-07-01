/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <unordered_map>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <wholegraph/tensor_description.h>

#include "error.hpp"

namespace wholegraph_tensor_ops {

struct one_wgt_hash {
  inline std::size_t operator()(const wholegraph_dtype_t& k) const
  {
    return static_cast<size_t>(k);
  }
};

struct two_wgt_hash {
  inline std::size_t operator()(const std::tuple<wholegraph_dtype_t, wholegraph_dtype_t>& k) const
  {
    return static_cast<size_t>(std::get<1>(k)) * (static_cast<size_t>(WHOLEGRAPH_DT_COUNT)) +
           static_cast<size_t>(std::get<0>(k));
  }
};

struct three_wgt_hash {
  inline std::size_t operator()(
    const std::tuple<wholegraph_dtype_t, wholegraph_dtype_t, wholegraph_dtype_t>& k) const
  {
    return static_cast<size_t>(std::get<2>(k)) * (static_cast<size_t>(WHOLEGRAPH_DT_COUNT)) *
             (static_cast<size_t>(WHOLEGRAPH_DT_COUNT)) +
           static_cast<size_t>(std::get<1>(k)) * (static_cast<size_t>(WHOLEGRAPH_DT_COUNT)) +
           static_cast<size_t>(std::get<0>(k));
  }
};

}  // namespace wholegraph_tensor_ops

template <typename DataTypeT>
inline wholegraph_dtype_t get_wholegraph_dtype()
{
  WHOLEGRAPH_FAIL_NOTHROW("get_wholegraph_dtype type not supported.");
  return WHOLEGRAPH_DT_UNKNOWN;
}

template <>
inline wholegraph_dtype_t get_wholegraph_dtype<int8_t>()
{
  return WHOLEGRAPH_DT_INT8;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<int16_t>()
{
  return WHOLEGRAPH_DT_INT16;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<int32_t>()
{
  return WHOLEGRAPH_DT_INT;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<int64_t>()
{
  return WHOLEGRAPH_DT_INT64;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<__half>()
{
  return WHOLEGRAPH_DT_HALF;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<__nv_bfloat16>()
{
  return WHOLEGRAPH_DT_BF16;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<float>()
{
  return WHOLEGRAPH_DT_FLOAT;
}
template <>
inline wholegraph_dtype_t get_wholegraph_dtype<double>()
{
  return WHOLEGRAPH_DT_DOUBLE;
}

#define VEC_SINT3264 std::vector<wholegraph_dtype_t>({WHOLEGRAPH_DT_INT, WHOLEGRAPH_DT_INT64})
#define VEC_ALLSINT                 \
  std::vector<wholegraph_dtype_t>( \
    {WHOLEGRAPH_DT_INT8, WHOLEGRAPH_DT_INT16, WHOLEGRAPH_DT_INT, WHOLEGRAPH_DT_INT64})

#define VEC_FLOAT_DOUBLE \
  std::vector<wholegraph_dtype_t>({WHOLEGRAPH_DT_FLOAT, WHOLEGRAPH_DT_DOUBLE})
#define VEC_HALF_FLOAT std::vector<wholegraph_dtype_t>({WHOLEGRAPH_DT_HALF, WHOLEGRAPH_DT_FLOAT})
#define VEC_BF16_HALF_FLOAT \
  std::vector<wholegraph_dtype_t>({WHOLEGRAPH_DT_BF16, WHOLEGRAPH_DT_HALF, WHOLEGRAPH_DT_FLOAT})
#define VEC_HALF_FLOAT_DOUBLE       \
  std::vector<wholegraph_dtype_t>( \
    {WHOLEGRAPH_DT_HALF, WHOLEGRAPH_DT_FLOAT, WHOLEGRAPH_DT_DOUBLE})
#define VEC_ALLFLOAT                \
  std::vector<wholegraph_dtype_t>( \
    {WHOLEGRAPH_DT_BF16, WHOLEGRAPH_DT_HALF, WHOLEGRAPH_DT_FLOAT, WHOLEGRAPH_DT_DOUBLE})
#define VEC_ALLSINT_ALLFLOAT                              \
  std::vector<wholegraph_dtype_t>({WHOLEGRAPH_DT_INT8,  \
                                    WHOLEGRAPH_DT_INT16, \
                                    WHOLEGRAPH_DT_INT,   \
                                    WHOLEGRAPH_DT_INT64, \
                                    WHOLEGRAPH_DT_BF16,  \
                                    WHOLEGRAPH_DT_HALF,  \
                                    WHOLEGRAPH_DT_FLOAT, \
                                    WHOLEGRAPH_DT_DOUBLE})

#define CASES_SINT3264(TEMPFUNC_NAME, ...)   \
  case WHOLEGRAPH_DT_INT: {                 \
    TEMPFUNC_NAME<int32_t, ##__VA_ARGS__>(); \
    break;                                   \
  }                                          \
  case WHOLEGRAPH_DT_INT64: {               \
    TEMPFUNC_NAME<int64_t, ##__VA_ARGS__>(); \
    break;                                   \
  }

#define CASES_ALLSINT(TEMPFUNC_NAME, ...)    \
  case WHOLEGRAPH_DT_INT8: {                \
    TEMPFUNC_NAME<int8_t, ##__VA_ARGS__>();  \
    break;                                   \
  }                                          \
  case WHOLEGRAPH_DT_INT16: {               \
    TEMPFUNC_NAME<int16_t, ##__VA_ARGS__>(); \
    break;                                   \
  }                                          \
    CASES_SINT3264(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
  case WHOLEGRAPH_DT_FLOAT: {                 \
    TEMPFUNC_NAME<float, ##__VA_ARGS__>();     \
    break;                                     \
  }                                            \
  case WHOLEGRAPH_DT_DOUBLE: {                \
    TEMPFUNC_NAME<double, ##__VA_ARGS__>();    \
    break;                                     \
  }

#define CASES_HALF_FLOAT(TEMPFUNC_NAME, ...) \
  case WHOLEGRAPH_DT_HALF: {                \
    TEMPFUNC_NAME<__half, ##__VA_ARGS__>();  \
    break;                                   \
  }                                          \
  case WHOLEGRAPH_DT_FLOAT: {               \
    TEMPFUNC_NAME<float, ##__VA_ARGS__>();   \
    break;                                   \
  }

#define CASES_BF16_HALF_FLOAT(TEMPFUNC_NAME, ...)  \
  case WHOLEGRAPH_DT_BF16: {                      \
    TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); \
    break;                                         \
  }                                                \
    CASES_HALF_FLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
  case WHOLEGRAPH_DT_HALF: {                       \
    TEMPFUNC_NAME<__half, ##__VA_ARGS__>();         \
    break;                                          \
  }                                                 \
    CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLFLOAT(TEMPFUNC_NAME, ...)         \
  case WHOLEGRAPH_DT_BF16: {                      \
    TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); \
    break;                                         \
  }                                                \
    CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLSINT_ALLFLOAT(TEMPFUNC_NAME, ...) \
  CASES_ALLSINT(TEMPFUNC_NAME, ##__VA_ARGS__)      \
  CASES_ALLFLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define REGISTER_DISPATCH_ONE_TYPE(NAME, TEMPFUNC_NAME, ARG0_SET)                           \
  static std::unordered_map<wholegraph_dtype_t,                                            \
                            decltype(&TEMPFUNC_NAME<int>),                                  \
                            wholegraph_tensor_ops::one_wgt_hash>* NAME##_dispatch1_map = nullptr; \
  template <typename T0>                                                                    \
  void Register##NAME##Map1FuncHelper0()                                                    \
  {                                                                                         \
    auto key = get_wholegraph_dtype<T0>();                                                 \
    NAME##_dispatch1_map->emplace(key, TEMPFUNC_NAME<T0>);                                  \
  }                                                                                         \
  __attribute__((constructor)) static void Register##NAME##Map1Func()                       \
  {                                                                                         \
    NAME##_dispatch1_map = new std::unordered_map<wholegraph_dtype_t,                      \
                                                  decltype(&TEMPFUNC_NAME<int>),            \
                                                  wholegraph_tensor_ops::one_wgt_hash>();         \
    auto arg0_types      = VEC_##ARG0_SET;                                                  \
    for (auto arg0_type : arg0_types) {                                                     \
      switch (arg0_type) {                                                                  \
        CASES_##ARG0_SET(Register##NAME##Map1FuncHelper0) default:                          \
        {                                                                                   \
          WHOLEGRAPH_FAIL_NOTHROW("dispatch with type=%d for function %s failed.",         \
                                   static_cast<int>(arg0_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

#define DISPATCH_ONE_TYPE(WGTypeValue0, NAME, ...)                \
  do {                                                            \
    auto key = WGTypeValue0;                                      \
    auto it  = NAME##_dispatch1_map->find(key);                   \
    WHOLEGRAPH_CHECK_NOTHROW(it != NAME##_dispatch1_map->end()); \
    it->second(__VA_ARGS__);                                      \
  } while (0)

#define REGISTER_DISPATCH_TWO_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET)                \
  static std::unordered_map<std::tuple<wholegraph_dtype_t, wholegraph_dtype_t>,           \
                            decltype(&TEMPFUNC_NAME<int, int>),                             \
                            wholegraph_tensor_ops::two_wgt_hash>* NAME##_dispatch2_map = nullptr; \
  template <typename T0, typename T1>                                                       \
  void Register##NAME##Map2FuncHelper0()                                                    \
  {                                                                                         \
    auto key = std::make_tuple(get_wholegraph_dtype<T0>(), get_wholegraph_dtype<T1>());   \
    NAME##_dispatch2_map->emplace(key, TEMPFUNC_NAME<T0, T1>);                              \
  }                                                                                         \
  template <typename T1>                                                                    \
  void Register##NAME##Map2FuncHelper1()                                                    \
  {                                                                                         \
    auto arg0_types = VEC_##ARG0_SET;                                                       \
    for (auto arg0_type : arg0_types) {                                                     \
      switch (arg0_type) {                                                                  \
        CASES_##ARG0_SET(Register##NAME##Map2FuncHelper0, T1) default:                      \
        {                                                                                   \
          WHOLEGRAPH_FAIL_NOTHROW("dispatch with type0=%d for function %s failed.",        \
                                   static_cast<int>(arg0_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }                                                                                         \
  __attribute__((constructor)) static void Register##NAME##Map2Func()                       \
  {                                                                                         \
    NAME##_dispatch2_map =                                                                  \
      new std::unordered_map<std::tuple<wholegraph_dtype_t, wholegraph_dtype_t>,          \
                             decltype(&TEMPFUNC_NAME<int, int>),                            \
                             wholegraph_tensor_ops::two_wgt_hash>();                              \
    auto arg1_types = VEC_##ARG1_SET;                                                       \
    for (auto arg1_type : arg1_types) {                                                     \
      switch (arg1_type) {                                                                  \
        CASES_##ARG1_SET(Register##NAME##Map2FuncHelper1) default:                          \
        {                                                                                   \
          WHOLEGRAPH_FAIL_NOTHROW("dispatch with type1=%d for function %s failed.",        \
                                   static_cast<int>(arg1_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

#define DISPATCH_TWO_TYPES(WGTypeValue0, WGTypeValue1, NAME, ...) \
  do {                                                            \
    auto key = std::make_tuple(WGTypeValue0, WGTypeValue1);       \
    auto it  = NAME##_dispatch2_map->find(key);                   \
    WHOLEGRAPH_CHECK_NOTHROW(it != NAME##_dispatch2_map->end()); \
    it->second(__VA_ARGS__);                                      \
  } while (0)

#define REGISTER_DISPATCH_THREE_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET, ARG2_SET)      \
  static std::unordered_map<                                                                  \
    std::tuple<wholegraph_dtype_t, wholegraph_dtype_t, wholegraph_dtype_t>,                \
    decltype(&TEMPFUNC_NAME<int, int, int>),                                                  \
    wholegraph_tensor_ops::three_wgt_hash>* NAME##_dispatch3_map = nullptr;                         \
  template <typename T0, typename T1, typename T2>                                            \
  void Register##NAME##Map3FuncHelper0()                                                      \
  {                                                                                           \
    auto key = std::make_tuple(                                                               \
      get_wholegraph_dtype<T0>(), get_wholegraph_dtype<T1>(), get_wholegraph_dtype<T2>()); \
    NAME##_dispatch3_map->emplace(key, TEMPFUNC_NAME<T0, T1, T2>);                            \
  }                                                                                           \
  template <typename T1, typename T2>                                                         \
  void Register##NAME##Map3FuncHelper1()                                                      \
  {                                                                                           \
    auto arg0_types = VEC_##ARG0_SET;                                                         \
    for (auto arg0_type : arg0_types) {                                                       \
      switch (arg0_type) {                                                                    \
        CASES_##ARG0_SET(Register##NAME##Map3FuncHelper0, T1, T2) default:                    \
        {                                                                                     \
          WHOLEGRAPH_FAIL_NOTHROW("dispatch with type0=%d for function %s failed.",          \
                                   static_cast<int>(arg0_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  template <typename T2>                                                                      \
  void Register##NAME##Map3FuncHelper2()                                                      \
  {                                                                                           \
    auto arg1_types = VEC_##ARG1_SET;                                                         \
    for (auto arg1_type : arg1_types) {                                                       \
      switch (arg1_type) {                                                                    \
        CASES_##ARG1_SET(Register##NAME##Map3FuncHelper1, T2) default:                        \
        {                                                                                     \
          WHOLEGRAPH_FAIL_NOTHROW("dispatch with type1=%d for function %s failed.",          \
                                   static_cast<int>(arg1_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  __attribute__((constructor)) static void Register##NAME##Map3Func()                         \
  {                                                                                           \
    NAME##_dispatch3_map = new std::unordered_map<                                            \
      std::tuple<wholegraph_dtype_t, wholegraph_dtype_t, wholegraph_dtype_t>,              \
      decltype(&TEMPFUNC_NAME<int, int, int>),                                                \
      wholegraph_tensor_ops::three_wgt_hash>();                                                     \
    auto arg2_types = VEC_##ARG2_SET;                                                         \
    for (auto arg2_type : arg2_types) {                                                       \
      switch (arg2_type) {                                                                    \
        CASES_##ARG2_SET(Register##NAME##Map3FuncHelper2) default:                            \
        {                                                                                     \
          WHOLEGRAPH_FAIL_NOTHROW("dispatch with type2=%d for function %s failed.",          \
                                   static_cast<int>(arg2_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }

#define DISPATCH_THREE_TYPES(WGTypeValue0, WGTypeValue1, WGTypeValue2, NAME, ...) \
  do {                                                                            \
    auto key = std::make_tuple(WGTypeValue0, WGTypeValue1, WGTypeValue2);         \
    auto it  = NAME##_dispatch3_map->find(key);                                   \
    WHOLEGRAPH_CHECK_NOTHROW(it != NAME##_dispatch3_map->end());                 \
    it->second(__VA_ARGS__);                                                      \
  } while (0)
