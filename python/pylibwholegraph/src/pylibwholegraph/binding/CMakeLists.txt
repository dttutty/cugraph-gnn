# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(linked_libraries wholegraph::wholegraph)
set(
  nanobind_sources
  binding_utils.cpp
  communicator_binding.cpp
  embedding_binding.cpp
  global_context_binding.cpp
  operations_binding.cpp
  registrations.cpp
  tensor_binding.cpp
  wholegraph_binding.cpp
)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
  REQUIRED
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(wholegraph_binding STABLE_ABI ${nanobind_sources})
target_link_libraries(wholegraph_binding PRIVATE ${linked_libraries})
set_target_properties(
  wholegraph_binding PROPERTIES INSTALL_RPATH "\$ORIGIN/../../libwholegraph/lib64"
)

install(TARGETS wholegraph_binding
  LIBRARY DESTINATION pylibwholegraph/binding
  RUNTIME DESTINATION pylibwholegraph/binding
)
