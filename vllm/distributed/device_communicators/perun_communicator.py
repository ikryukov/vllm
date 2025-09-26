# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.utils import current_stream

try:
    import perun
    perun_available = True
except ImportError:
    perun_available = False

logger = logging.getLogger(__name__)


class PerunCommunicator:

    def __init__(self, group: ProcessGroup,
                 device: Union[int, str, torch.device]) -> None:
        self.disabled = True

        if not perun_available:
            logger.warning(
                "Perun module not available. Perun communicator disabled.")
            return

        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)

        self._comm = perun.comm_create(self.world_size, self.rank)

        # Get the current CUDA stream for the current device
        if torch.cuda.is_available():
            # Ensure we're using the current device and get its stream
            self._device = torch.cuda.current_device()
            # self._stream = torch.cuda.current_stream(device=self._device)
            self._stream = current_stream()
        else:
            raise RuntimeError("CUDA is not available")

        # Enable perun communicator if initialization succeeds
        self.disabled = False

    def _get_current_stream(self):
        """Get the current CUDA stream for the current device"""
        # TODO: in vLLM use theirs implementaion
        # return torch.cuda.current_stream(device=self._device)
        return self._stream

    def __del__(self):
        try:
            if hasattr(self, "_comm") and self._comm and perun_available:
                perun.comm_destroy(self._comm)
        except Exception:
            pass

    def should_use_perun_allreduce(self, tensor: torch.Tensor) -> bool:
        """
        Determine if Perun allreduce should be used for the given tensor.

        Args:
            tensor: The tensor to check.

        Returns:
            True if Perun allreduce should be used, False otherwise.
        """
        # Add any additional tensor-specific checks here
        return not self.disabled

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, list[torch.Tensor]],
        dim: int = 0,
        sizes: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")

        if isinstance(input_, list):
            local = torch.cat([t.contiguous() for t in input_], dim=0)
        else:
            local = input_.contiguous()

        if local.dim() != 1:
            raise NotImplementedError("test only supports 1D tensors")

        itemsize = local.element_size()

        if sizes is None:
            raise ValueError("sizes must be provided for all_gatherv test")
        if len(sizes) != self.world_size:
            raise ValueError("sizes length must equal world size")

        recvcounts_bytes = [int(s) * itemsize for s in sizes]
        displs_bytes: list[int] = []
        running = 0
        for c in recvcounts_bytes:
            displs_bytes.append(running)
            running += int(c)

        total_elems = sum(sizes)
        out = torch.empty(total_elems, dtype=local.dtype, device=local.device)

        # Build contiguous uint32 arrays for counts/displacements
        # and pass pointers
        rc = np.asarray(recvcounts_bytes, dtype=np.uint32)
        dp = np.asarray(displs_bytes, dtype=np.uint32)

        perun.allgatherv(
            self._comm,
            self._get_current_stream().cuda_stream,
            local.data_ptr(),
            int(local.numel() * itemsize),
            out.data_ptr(),
            rc.ctypes.data,
            dp.ctypes.data,
        )

        return out

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None):
        world_size = self.world_size
        assert self._comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        perun.reducescatterv(self._comm,
                             self._get_current_stream().cuda_stream,
                             input_tensor.data_ptr(), input_tensor.numel(),
                             output.data_ptr(), sizes, input_tensor.dtype,
                             torch.distributed.ReduceOp.SUM)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_reduce(self, input_):
        perun.allreduce(self._comm,
                        self._get_current_stream().cuda_stream,
                        input_.data_ptr(), input_.numel(), input_.data_ptr(),
                        input_.dtype, torch.distributed.ReduceOp.SUM)
        return input_
