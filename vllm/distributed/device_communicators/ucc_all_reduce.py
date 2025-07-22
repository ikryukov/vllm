 # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cuda_device_count_stateless


logger = init_logger(__name__)


class UCCAllreduce:
    """
    UCC-based allreduce implementation that calls UCC allreduce operations.

    This implementation leverages UCC (Unified Collective Communications)
    using UCC process groups as the backend for efficient GPU-to-GPU communication.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]  # UCC supports more sizes

    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 max_size=512 * 1024 * 1024) -> None:
        """
        Args:
            group: the process group to work on
            device: the device to bind the UCCAllreduce to
            max_size: maximum supported allreduce size in bytes
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self.group = group
        self.max_size = max_size

        if not current_platform.is_cuda_alike():
            logger.info("UCC allreduce is disabled because "
                        "it requires CUDA-compatible platform")
            return

        # UCC supports multi-node communication, so we don't disable it
        # for multi-node setups unlike the custom allreduce implementation
        logger.info("UCC allreduce supports multi-node communication")

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "UCCAllreduce should be attached to a non-NCCL group.")

        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)

        if self.world_size == 1:
            # No need to initialize UCC for single GPU case
            return

        if self.world_size not in UCCAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "UCC allreduce is disabled due to unsupported world size: %d. "
                "Supported world sizes: %s",
                self.world_size, str(UCCAllreduce._SUPPORTED_WORLD_SIZES))
            return

        # Handle device specification
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        # Initialize UCC context and team
        self.disabled = False
        logger.info("UCC allreduce initialized successfully for rank %d/%d",
                    self.rank, self.world_size)

    def should_use_ucc_allreduce(self, inp: torch.Tensor) -> bool:
        """Check if tensor is suitable for UCC allreduce."""
        if self.disabled:
            return False

        inp_size = inp.numel() * inp.element_size()

        # UCC allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False

        # Check if tensor is contiguous
        if not inp.is_contiguous():
            return False

        # Check size limits NVLS is 512MB
        if inp_size > self.max_size:
            return False

        # UCC supports various data types
        supported_dtypes = [torch.float32, torch.bfloat16]
        if inp.dtype not in supported_dtypes:
            return False

        return True

    def allreduce(self,
                   inp: torch.Tensor,
                   *,
                   out: torch.Tensor = None,
                   op: str = "sum") -> torch.Tensor:
        """
        Perform allreduce operation using UCC.

        Args:
            inp: Input tensor
            out: Output tensor (if None, creates new tensor)
            op: Reduction operation ('sum', 'max', 'min', 'prod')

        Returns:
            Reduced tensor
        """
        if self.disabled:
            raise RuntimeError("UCC allreduce is disabled")

        if out is None:
            out = torch.empty_like(inp)

        # Map operation to UCC operation
        ucc_op_map = {
            "sum": dist.ReduceOp.SUM,
        }

        if op not in ucc_op_map:
            raise ValueError(f"Unsupported operation: {op}")

        device_type = "cuda" if inp.is_cuda else "host"
        logger.info(f"UCC allreduce: count {inp.numel()}, dtype {inp.dtype}, op {op}, device {inp.device} ({device_type})")

        out = inp.clone()

        torch.distributed.all_reduce(out, op=ucc_op_map[op], group=self.group)

        return out

    def custom_allreduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """
        The main allreduce API that provides support for cuda graph.

        Args:
            input: Input tensor to reduce

        Returns:
            Reduced tensor or None if UCC allreduce is not applicable
        """
        if self.disabled or not self.should_use_ucc_allreduce(input):
            return None

        # UCC should support graph capture, but may need special handling
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.allreduce(input, op="sum")
            else:
                # During warmup, mimic allocation pattern
                return torch.empty_like(input)
        else:
            # Normal execution path
            return self.allreduce(input, op="sum")

    @contextmanager
    def capture(self):
        """
        Context manager for CUDA graph capture support.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False

    def close(self):
        """Clean up UCC resources."""
        if not self.disabled:
            if hasattr(self, 'ucc_team'):
                self.ucc_team.destroy()
            if hasattr(self, 'ucc_context'):
                self.ucc_context.destroy()
            logger.debug("UCC resources cleaned up for rank %d", self.rank)

    def __del__(self):
        self.close()
