# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pylibwholegraph.binding.wholegraph_binding as wgb
from pylibwholegraph.utils.imports import import_optional
from .utils import (
    torch_dtype_to_wholegraph_dtype,
    wholegraph_dtype_to_torch_dtype,
    get_file_size,
)
from .utils import str_to_wholegraph_memory_type, str_to_wholegraph_location
from .utils import get_part_file_name, get_part_file_list
from .comm import WholeGraphCommunicator
from typing import Union, List
from .dlpack_utils import torch_import_from_dlpack
from .wholegraph_env import wrap_torch_tensor, get_wholegraph_env_fns, get_stream

torch = import_optional("torch")

WholeGraphMemoryType = wgb.WholeGraphMemoryType
WholeGraphMemoryLocation = wgb.WholeGraphMemoryLocation


class WholeGraphTensor(object):
    r"""WholeGraph tensor"""

    def __init__(self, wgb_tensor: wgb.PyWholeGraphTensor):
        self.wgb_tensor = wgb_tensor

    @property
    def dtype(self):
        return wholegraph_dtype_to_torch_dtype(self.wgb_tensor.dtype)

    def dim(self):
        return self.wgb_tensor.dim()

    @property
    def shape(self):
        return self.wgb_tensor.shape

    def stride(self):
        return self.wgb_tensor.stride()

    def storage_offset(self):
        return self.wgb_tensor.storage_offset()

    def get_comm(self):
        return WholeGraphCommunicator(
            self.wgb_tensor.get_wholegraph_handle().get_communicator()
        )

    def gather(
        self, indice: "torch.Tensor", *, force_dtype: Union["torch.dtype", None] = None
    ):
        assert indice.dim() == 1
        embedding_dim = self.shape[1] if self.dim() == 2 else 1
        embedding_count = indice.shape[0]
        current_cuda_device = "cuda:%d" % (torch.cuda.current_device(),)
        output_dtype = force_dtype if force_dtype is not None else self.dtype
        output_tensor = torch.empty(
            [embedding_count, embedding_dim],
            device=current_cuda_device,
            dtype=output_dtype,
            requires_grad=False,
        )
        wgb.wholegraph_gather_op(
            self.wgb_tensor,
            wrap_torch_tensor(indice),
            wrap_torch_tensor(output_tensor),
            get_wholegraph_env_fns(),
            get_stream(),
        )
        return output_tensor.view(-1) if self.dim() == 1 else output_tensor

    def scatter(self, input_tensor: "torch.Tensor", indice: "torch.Tensor"):
        assert indice.dim() == 1
        assert input_tensor.dim() == self.dim()
        assert indice.shape[0] == input_tensor.shape[0]
        if self.dim() == 2:
            assert input_tensor.shape[1] == self.shape[1]
        else:
            # unsqueeze to 2D tensor because wgb_tensor is unsqueezed within scatter_op
            input_tensor = input_tensor.unsqueeze(1)
        wgb.wholegraph_scatter_op(
            wrap_torch_tensor(input_tensor),
            wrap_torch_tensor(indice),
            self.wgb_tensor,
            get_wholegraph_env_fns(),
            get_stream(),
        )

    def get_sub_tensor(self, starts, ends):
        """
        Get sub tensor of WholeGraph tensor
        :param starts: An array of the start indices of each dim
        :param ends: An array of the end indices of each dim, -1 means
          to the last element
        :return: WholeGraph tensor
        """
        return WholeGraphTensor(self.wgb_tensor.get_sub_tensor(starts, ends))

    def get_local_tensor(self, host_view: bool = False):
        """Get local tensor of WholeGraph tensor
        :param host_view: Get host view or not, if True, return host tensor,
          else return device tensor
        :return: Tuple of DLPack Tensor and element offset.
        """
        if host_view:
            return self.wgb_tensor.get_local_tensor(
                torch_import_from_dlpack, WholeGraphMemoryLocation.MlHost, -1
            )
        else:
            return self.wgb_tensor.get_local_tensor(
                torch_import_from_dlpack,
                WholeGraphMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def get_global_tensor(self, host_view: bool = False):
        """Get global tensor of WholeGraph tensor
        :param host_view: Get host view or not, if True, return host tensor,
          else return device tensor
        :return: Tuple of DLPack Tensor and element offset (0 for global tensor).
        """
        if host_view:
            return self.wgb_tensor.get_global_tensor(
                torch_import_from_dlpack, WholeGraphMemoryLocation.MlHost, -1
            )
        else:
            return self.wgb_tensor.get_global_tensor(
                torch_import_from_dlpack,
                WholeGraphMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def from_filelist(self, filelist: Union[List[str], str], round_robin_size: int = 0):
        """
        Load WholeGraph tensor from file lists
        :param filelist: file list to load from
        :param round_robin_size: continuous embedding size of a rank
          using round robin shard strategy
        :return: None
        """
        if isinstance(filelist, str):
            filelist = [filelist]
        self.wgb_tensor.from_filelist(filelist, round_robin_size)

    def from_file_prefix(self, file_prefix: str, part_count: Union[int, None] = None):
        """
        Load WholeGraph tensor from files with same prefix, files has format
            "%s_part_%d_of_%d" % (prefix, part_id, part_count)
        :param file_prefix: file name prefix
        :param part_count: part count of file
        :return: None
        """
        if part_count is None:
            part_count = self.get_comm().get_size()
        file_list = get_part_file_list(file_prefix, part_count)
        self.from_filelist(file_list)

    def local_to_file(self, filename: str):
        """
        Store local tensor of WholeGraph tensor to file, all ranks should
          call this together with different filename
        :param filename: file name of local tensor file.
        :return: None
        """
        self.wgb_tensor.to_file(filename)

    def to_file_prefix(self, file_prefix: str):
        """
        Store WholeGraph tensor to files with same prefix.
        :param file_prefix: file name prefix
        :return: None
        """
        wg_comm = self.get_comm()
        filename = get_part_file_name(
            file_prefix, wg_comm.get_rank(), wg_comm.get_size()
        )
        self.local_to_file(filename)


def create_wholegraph_tensor(
    comm: WholeGraphCommunicator,
    memory_type: str,
    memory_location: str,
    sizes: List[int],
    dtype: "torch.dtype",
    strides: List[int],
    tensor_entry_partition: Union[List[int], None] = None,
):
    """
    Create empty WholeGraph tensor. Now only support dim = 1 or 2
    :param comm: WholeGraphCommunicator
    :param memory_type: WholeGraph memory type, should be continuous or distributed
    :param memory_location: WholeGraph memory location, should be cpu or cuda
    :param sizes: size of the tensor
    :param dtype: data type of the tensor
    :param strides: strides of the tensor
    :param tensor_entry_partition: rank partition based on entry;
      tensor_entry_partition[i] determines the entry count of rank
      i and shoud be a positive integer; the sum of tensor_entry_partition
      should equal to total entry count; entries will be equally partitioned if None
    :return: Allocated WholeGraphTensor
    """
    dim = len(sizes)
    if dim < 1 or dim > 2:
        raise ValueError("Only dim 1 or 2 is supported now.")
    if strides is None:
        strides = [1] * dim
        strides[0] = sizes[1] if dim == 2 else 1
    else:
        assert len(strides) == dim
        assert strides[-1] == 1
        if dim == 2:
            assert strides[0] >= sizes[1]
    td = wgb.PyWholeGraphTensorDescription()
    td.set_shape(sizes)
    td.set_stride(strides)
    td.set_dtype(torch_dtype_to_wholegraph_dtype(dtype))

    wg_memory_type = str_to_wholegraph_memory_type(memory_type)
    wg_location = str_to_wholegraph_location(memory_location)

    return WholeGraphTensor(
        wgb.create_wholegraph_tensor(
            td, comm.wgb_comm, wg_memory_type, wg_location, tensor_entry_partition
        )
    )


def create_wholegraph_tensor_from_filelist(
    comm: WholeGraphCommunicator,
    memory_type: str,
    memory_location: str,
    filelist: Union[List[str], str],
    dtype: "torch.dtype",
    last_dim_size: int = 0,
    last_dim_strides: int = -1,
    tensor_entry_partition: Union[List[int], None] = None,
):
    """
    Create WholeGraph tensor from list of binary files.
    :param comm: WholeGraphCommunicator
    :param memory_type: WholeGraph memory type, should be continuous or distributed
    :param memory_location: WholeGraph memory location, should be cpu or cuda
    :param filelist: list of binary files
    :param dtype: data type of the tensor
    :param last_dim_size: 0 for create 1-D array, positive value for
      create matrix column size
    :param last_dim_strides: stride of last_dim, -1 for same as size of last dim.
    :param tensor_entry_partition: rank partition based on entry;
      tensor_entry_partition[i] determines the entry count of rank
      i and shoud be a positive integer; the sum of tensor_entry_partition
      should equal to total entry count; entries will be equally partitioned if None
    :return: WholeGraphTensor
    """
    if isinstance(filelist, str):
        filelist = [filelist]
    element_size = torch.tensor([], dtype=dtype).element_size()
    if last_dim_strides == -1:
        last_dim_strides = last_dim_size if last_dim_size > 0 else 1
    file_entry_size = (
        element_size * last_dim_size if last_dim_size > 0 else element_size
    )
    total_file_size = 0
    for filename in filelist:
        file_size = get_file_size(filename)
        if file_size % file_entry_size != 0:
            raise ValueError(
                "File %s size is %d not mutlple of %d"
                % (filename, file_size, file_entry_size)
            )
        total_file_size += file_size
    total_entry_count = total_file_size // file_entry_size
    if last_dim_size == 0:
        sizes = [total_entry_count]
        strides = [1]
    else:
        sizes = [total_entry_count, last_dim_size]
        strides = [last_dim_strides, 1]
    wg_tensor = create_wholegraph_tensor(
        comm,
        memory_type,
        memory_location,
        sizes,
        dtype,
        strides,
        tensor_entry_partition,
    )
    wg_tensor.from_filelist(filelist)
    return wg_tensor


def destroy_wholegraph_tensor(wg_tensor: WholeGraphTensor):
    """
    Destroy allocated WholeGraph tensor
    :param wg_tensor: WholeGraph tensor
    :return: None
    """
    wgb.destroy_wholegraph_tensor(wg_tensor.wgb_tensor)
    wg_tensor.wgb_tensor = None
