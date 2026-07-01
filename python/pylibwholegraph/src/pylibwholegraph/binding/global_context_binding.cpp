/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "global_context_binding.hpp"

nb::object tensor_shape_tuple(wholegraph_tensor_description_t const* tensor_desc)
{
  if (tensor_desc == nullptr || tensor_desc->dim < 0 ||
      (tensor_desc->dim > 0 && tensor_desc->sizes == nullptr)) {
    throw nb::value_error("Invalid tensor description");
  }
  return int64_tuple_from_values(tensor_desc->sizes, tensor_desc->dim);
}

void discard_pending_callback_error(char const* callback_name)
{
  if (!PyErr_Occurred()) { PyErr_Format(PyExc_RuntimeError, "%s failed", callback_name); }
  nb::python_error error;
  error.discard_as_unraisable(callback_name);
}

bool ensure_callback_initialized(bool initialized, char const* callback_name) noexcept
{
  if (initialized) { return true; }
  PyErr_SetString(PyExc_RuntimeError, "GlobalContextWrapper is not initialized");
  discard_pending_callback_error(callback_name);
  return false;
}

PyObject* python_builtin_exception_type(nb::exception_type type)
{
  switch (type) {
    case nb::exception_type::stop_iteration: return PyExc_StopIteration;
    case nb::exception_type::index_error: return PyExc_IndexError;
    case nb::exception_type::key_error: return PyExc_KeyError;
    case nb::exception_type::value_error: return PyExc_ValueError;
    case nb::exception_type::type_error: return PyExc_TypeError;
    case nb::exception_type::buffer_error: return PyExc_BufferError;
    case nb::exception_type::import_error: return PyExc_ImportError;
    case nb::exception_type::attribute_error: return PyExc_AttributeError;
    case nb::exception_type::runtime_error:
    case nb::exception_type::next_overload: return PyExc_RuntimeError;
  }
  return PyExc_RuntimeError;
}

void set_builtin_exception(nb::builtin_exception const& error)
{
  PyErr_SetString(python_builtin_exception_type(error.type()), error.what());
}

void release_python_memory_context(void* memory_context)
{
  if (memory_context != nullptr) { nb::handle(static_cast<PyObject*>(memory_context)).dec_ref(); }
}

template <typename... Args>
nb::object call_python_callback(char const* callback_name,
                                PyObject* callback,
                                Args&&... args) noexcept
{
  try {
    return nb::borrow<nb::object>(callback)(std::forward<Args>(args)...);
  } catch (nb::python_error& error) {
    error.discard_as_unraisable(callback_name);
    return {};
  } catch (nb::builtin_exception const& error) {
    set_builtin_exception(error);
  } catch (std::exception const& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown Python callback error");
  }
  discard_pending_callback_error(callback_name);
  return {};
}

void python_cb_wrapper_temp_create_context(void** memory_context, void* global_context) noexcept
{
  if (memory_context == nullptr) { return; }
  *memory_context = nullptr;

  nb::gil_scoped_acquire gil;
  auto* wrapper              = static_cast<GlobalContextWrapperNB*>(global_context);
  if (!ensure_callback_initialized(wrapper != nullptr && wrapper->temp_create_context_fn_.is_valid(),
                                   "temporary create_context callback")) {
    return;
  }

  nb::object result = call_python_callback("temporary create_context callback",
                                           wrapper->temp_create_context_fn_.ptr(),
                                           nb::handle(wrapper->temp_global_context()));
  if (result.is_valid()) { *memory_context = result.release().ptr(); }
}

void python_cb_wrapper_temp_destroy_context(void* memory_context, void* global_context) noexcept
{
  nb::gil_scoped_acquire gil;
  auto* wrapper              = static_cast<GlobalContextWrapperNB*>(global_context);
  if (!ensure_callback_initialized(wrapper != nullptr && wrapper->temp_destroy_context_fn_.is_valid(),
                                   "temporary destroy_context callback")) {
    release_python_memory_context(memory_context);
    return;
  }

  PyObject* mem_ctx = GlobalContextWrapperNB::memory_context_object(memory_context);
  (void)call_python_callback("temporary destroy_context callback",
                             wrapper->temp_destroy_context_fn_.ptr(),
                             nb::handle(mem_ctx),
                             nb::handle(wrapper->temp_global_context()));
  release_python_memory_context(memory_context);
}

void* call_python_malloc_callback(PyObject* callback,
                                  PyObject* global_context,
                                  wholegraph_tensor_description_t* tensor_desc,
                                  wholegraph_memory_allocation_type_t malloc_type,
                                  void* memory_context,
                                  char const* callback_name) noexcept
{
  try {
    if (callback == nullptr) {
      ensure_callback_initialized(false, callback_name);
      return nullptr;
    }

    nb::object shape = tensor_shape_tuple(tensor_desc);
    nb::int_ dtype{static_cast<long>(tensor_desc->dtype)};
    nb::int_ malloc_type_obj{static_cast<long>(malloc_type)};

    PyObject* mem_ctx = GlobalContextWrapperNB::memory_context_object(memory_context);
    nb::object result = call_python_callback(callback_name,
                                             callback,
                                             shape,
                                             dtype,
                                             malloc_type_obj,
                                             nb::handle(mem_ctx),
                                             nb::handle(global_context));
    if (!result.is_valid()) { return nullptr; }

    return reinterpret_cast<void*>(
      static_cast<uintptr_t>(nb::cast<int64_t>(result)));
  } catch (nb::python_error& error) {
    error.discard_as_unraisable(callback_name);
    return nullptr;
  } catch (nb::builtin_exception const& error) {
    set_builtin_exception(error);
  } catch (nb::cast_error const&) {
    PyErr_SetString(PyExc_TypeError, "Python malloc callback must return an integer pointer");
  } catch (std::exception const& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown Python malloc callback error");
  }
  discard_pending_callback_error(callback_name);
  return nullptr;
}

void* python_cb_wrapper_temp_malloc(wholegraph_tensor_description_t* tensor_desc,
                                    wholegraph_memory_allocation_type_t malloc_type,
                                    void* memory_context,
                                    void* global_context) noexcept
{
  nb::gil_scoped_acquire gil;
  auto* wrapper              = static_cast<GlobalContextWrapperNB*>(global_context);
  void* result               = nullptr;
  if (ensure_callback_initialized(wrapper != nullptr && wrapper->temp_malloc_fn_.is_valid(),
                                  "temporary malloc callback")) {
    result = call_python_malloc_callback(wrapper->temp_malloc_fn_.ptr(),
                                         wrapper->temp_global_context(),
                                         tensor_desc,
                                         malloc_type,
                                         memory_context,
                                         "temporary malloc callback");
  }
  return result;
}

void python_cb_wrapper_temp_free(void* memory_context, void* global_context) noexcept
{
  nb::gil_scoped_acquire gil;
  auto* wrapper              = static_cast<GlobalContextWrapperNB*>(global_context);
  if (!ensure_callback_initialized(wrapper != nullptr && wrapper->temp_free_fn_.is_valid(),
                                   "temporary free callback")) {
    return;
  }

  (void)call_python_callback(
    "temporary free callback",
    wrapper->temp_free_fn_.ptr(),
    nb::handle(GlobalContextWrapperNB::memory_context_object(memory_context)),
    nb::handle(wrapper->temp_global_context()));
}

void* python_cb_wrapper_output_malloc(wholegraph_tensor_description_t* tensor_desc,
                                      wholegraph_memory_allocation_type_t malloc_type,
                                      void* memory_context,
                                      void* global_context) noexcept
{
  nb::gil_scoped_acquire gil;
  auto* wrapper              = static_cast<GlobalContextWrapperNB*>(global_context);
  void* result               = nullptr;
  if (ensure_callback_initialized(wrapper != nullptr && wrapper->output_malloc_fn_.is_valid(),
                                  "output malloc callback")) {
    result = call_python_malloc_callback(wrapper->output_malloc_fn_.ptr(),
                                         wrapper->output_global_context(),
                                         tensor_desc,
                                         malloc_type,
                                         memory_context,
                                         "output malloc callback");
  }
  return result;
}

void python_cb_wrapper_output_free(void* memory_context, void* global_context) noexcept
{
  nb::gil_scoped_acquire gil;
  auto* wrapper              = static_cast<GlobalContextWrapperNB*>(global_context);
  if (!ensure_callback_initialized(wrapper != nullptr && wrapper->output_free_fn_.is_valid(),
                                   "output free callback")) {
    return;
  }

  (void)call_python_callback(
    "output free callback",
    wrapper->output_free_fn_.ptr(),
    nb::handle(GlobalContextWrapperNB::memory_context_object(memory_context)),
    nb::handle(wrapper->output_global_context()));
}
