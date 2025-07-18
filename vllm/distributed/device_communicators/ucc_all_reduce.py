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

try:
    import ucc
    ucc_available = True
except ImportError:
    ucc_available = False

logger = init_logger(__name__)


class UCCAllreduce:
    """
    UCC-based allreduce implementation that calls UCC allreduce operations.

    This implementation leverages UCC (Unified Collective Communications)
    using UCC process groups as the backend for efficient GPU-to-GPU communication.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16, 32, 64, 128]  # UCC supports more sizes

    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 max_size=8192 * 1024) -> None:
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

        if not ucc_available:
            logger.info("UCC allreduce is disabled because "
                        "UCC library is not available")
            return

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

        try:
            # Initialize UCC context and team
            self._init_ucc_context()
            self.disabled = False
            logger.info("UCC allreduce initialized successfully for rank %d/%d",
                        self.rank, self.world_size)
        except Exception as e:
            logger.warning("Failed to initialize UCC allreduce: %s", str(e))
            self.disabled = True

    def _init_ucc_context(self):
        """Initialize UCC context and team for allreduce operations."""
        # Get physical device IDs for UCC initialization
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(cuda_device_count_stateless()))

        physical_device_id = device_ids[self.device.index]

        # Initialize UCC context
        ucc_params = {
            'team_size': self.world_size,
            'rank': self.rank,
            'device_id': physical_device_id,
            'device_type': 'cuda',
            'lib_path': None,  # Use default UCC library path
        }

        # Create UCC context
        self.ucc_context = ucc.Context(**ucc_params)

        # Create team for collective operations
        self.ucc_team = self.ucc_context.create_team(
            size=self.world_size,
            rank=self.rank,
            oob_allgather=self._oob_allgather_callback
        )

        logger.debug("UCC context and team initialized for rank %d", self.rank)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
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

        # Check size limits
        if inp_size > self.max_size:
            return False

        # UCC supports various data types
        supported_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        if inp.dtype not in supported_dtypes:
            return False

        return True

    def all_reduce(self,
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
            "sum": ucc.ReduceOp.SUM,
            "max": ucc.ReduceOp.MAX,
            "min": ucc.ReduceOp.MIN,
            "prod": ucc.ReduceOp.PROD
        }

        if op not in ucc_op_map:
            raise ValueError(f"Unsupported operation: {op}")

        logger.info(f"UCC allreduce: count {inp.numel()}, dtype {inp.dtype}, op {op}")
        # Perform UCC allreduce
        request = self.ucc_team.allreduce(
            src=inp.data_ptr(),
            dst=out.data_ptr(),
            count=inp.numel(),
            dtype=self._torch_dtype_to_ucc(inp.dtype),
            op=ucc_op_map[op],
            stream=torch.cuda.current_stream().cuda_stream
        )

        # Wait for completion
        while not request.is_complete():
            self.ucc_context.progress()

        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """
        The main allreduce API that provides support for cuda graph.

        Args:
            input: Input tensor to reduce

        Returns:
            Reduced tensor or None if UCC allreduce is not applicable
        """
        if self.disabled or not self.should_custom_ar(input):
            return None

        # UCC should support graph capture, but may need special handling
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, op="sum")
            else:
                # During warmup, mimic allocation pattern
                return torch.empty_like(input)
        else:
            # Normal execution path
            return self.all_reduce(input, op="sum")

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

    def _oob_allgather_callback(self, req):
        """Out-of-band allgather callback for UCC team creation."""
        # This is used by UCC to exchange connection information
        # between processes during team creation
        tensor = torch.tensor([self.rank], dtype=torch.int, device="cpu")
        gather_list = [torch.tensor([0], dtype=torch.int, device="cpu")
                      for _ in range(self.world_size)]
        dist.all_gather(gather_list, tensor, group=self.group)
        return [t.item() for t in gather_list]

    def _torch_dtype_to_ucc(self, dtype: torch.dtype):
        """Convert PyTorch dtype to UCC dtype."""
        dtype_map = {
            torch.float32: ucc.DataType.FLOAT32,
            torch.float16: ucc.DataType.FLOAT16,
            torch.bfloat16: ucc.DataType.BFLOAT16,
            torch.int32: ucc.DataType.INT32,
            torch.int64: ucc.DataType.INT64,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype for UCC: {dtype}")

        return dtype_map[dtype]


# Utility functions for UCC detection and setup
def is_ucc_available() -> bool:
    """Check if UCC is available and properly configured."""
    return ucc_available


def get_ucc_version() -> Optional[str]:
    """Get UCC version if available."""
    if not ucc_available:
        return None
    try:
        return ucc.__version__
    except AttributeError:
        return "unknown"


def check_ucc_cuda_support() -> bool:
    """Check if UCC CUDA support is available."""
    if not ucc_available:
        return False

    try:
        # Check if UCC was compiled with CUDA support
        context = ucc.Context()
        has_cuda = hasattr(context, 'cuda_stream')
        context.destroy()
        return has_cuda
    except Exception:
        return False