# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

try:
    import perun

    perun_available = True
except ImportError:
    perun_available = False

logger = init_logger(__name__)


class PerunCommunicator:
    def __init__(self, group: ProcessGroup, device: int | str | torch.device) -> None:
        self.disabled = True
        self._output_buffer_cache: dict[
            tuple[int, torch.dtype, torch.device], torch.Tensor
        ] = {}
        # Statistics for all_reduce
        self._allreduce_shapes_count: defaultdict[tuple[int, ...], int] = defaultdict(
            int
        )
        # Statistics for all_gatherv
        self._allgatherv_shapes_count: defaultdict[tuple[int, ...], int] = defaultdict(
            int
        )
        # Statistics for reduce_scatterv
        self._reducescatterv_shapes_count: defaultdict[tuple[int, ...], int] = (
            defaultdict(int)
        )

        if not perun_available:
            logger.warning("Perun module not available. Perun communicator disabled.")
            return

        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)

        # Get the current CUDA stream for the current device
        if torch.cuda.is_available():
            # Ensure we're using the current device and get its stream
            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            elif isinstance(device, str):
                device = torch.device(device)
            assert isinstance(device, torch.device)
            self._device = device
            self._stream = current_stream()
            logger.info(
                "Perun communicator device: %s, stream: %s",
                self._device,
                self._stream,
            )
        else:
            logger.warning("CUDA is not available. Perun communicator disabled.")
            return

        self._comm = perun.comm_create(self.world_size, self.rank, self._device.index)
        # Enable perun communicator if initialization succeeds
        self.disabled = False
        logger.info(
            "Perun communicator initialized: rank=%d/%d, device=%s",
            self.rank,
            self.world_size,
            self._device,
        )

        # Register atexit handler to ensure statistics are printed on shutdown
        # This ensures stats are printed even if __del__ isn't called
        atexit.register(self._atexit_handler)

    def _get_current_stream(self):
        """Get the current CUDA stream for the current device"""
        # TODO: in vLLM use theirs implementaion
        # return torch.cuda.current_stream(device=self._device)
        return self._stream

    def _atexit_handler(self):
        """Handler called at process exit to ensure statistics are printed."""
        try:
            # Use print() to ensure output is visible even if logging fails
            if hasattr(self, "rank") and self.rank == 0:
                print(
                    f"\n[Perun] atexit handler called for rank {self.rank}", flush=True
                )
            self.print_statistics()
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            print(f"[Perun] Error in atexit handler: {e}", flush=True)

    def _log(self, msg, *args, use_print=False):
        """Helper to log or print messages."""
        if use_print:
            if args:
                print(msg % args, flush=True)
            else:
                print(msg, flush=True)
        else:
            logger.info(msg, *args)

    def print_statistics(self, use_print=True):
        """Print communication statistics summary.

        Args:
            use_print: If True, use print() for output (more reliable during shutdown).
                      If False, use logger.info().
        """
        if not hasattr(self, "rank"):
            return

        # Only print from rank 0 to avoid duplicate output
        if self.rank != 0:
            return

        has_stats = False

        # Check if we have any statistics to print
        if hasattr(self, "_allreduce_shapes_count") and self._allreduce_shapes_count:
            has_stats = True
        if hasattr(self, "_allgatherv_shapes_count") and self._allgatherv_shapes_count:
            has_stats = True
        if (
            hasattr(self, "_reducescatterv_shapes_count")
            and self._reducescatterv_shapes_count
        ):
            has_stats = True

        if not has_stats:
            self._log(
                "No Perun communication statistics to report", use_print=use_print
            )
            return

        self._log("=" * 80, use_print=use_print)
        self._log("Perun Communicator Statistics Summary (Rank 0)", use_print=use_print)
        self._log("=" * 80, use_print=use_print)

        # Print all_reduce statistics
        if hasattr(self, "_allreduce_shapes_count") and self._allreduce_shapes_count:
            sorted_shapes = sorted(
                self._allreduce_shapes_count.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            total_calls = sum(self._allreduce_shapes_count.values())
            self._log("ALL_REDUCE:", use_print=use_print)
            self._log("  Total calls: %d", total_calls, use_print=use_print)
            self._log(
                "  Unique tensor shapes: %d",
                len(self._allreduce_shapes_count),
                use_print=use_print,
            )
            for shape, count in sorted_shapes[:5]:  # Top 5
                percentage = (count / total_calls) * 100
                numel = np.prod(shape) if shape else 0
                self._log(
                    "    Shape %s: %d calls (%.2f%%), %s elements",
                    shape,
                    count,
                    percentage,
                    format(numel, ","),
                    use_print=use_print,
                )
            self._log("-" * 80, use_print=use_print)

        # Print all_gatherv statistics
        if hasattr(self, "_allgatherv_shapes_count") and self._allgatherv_shapes_count:
            sorted_shapes = sorted(
                self._allgatherv_shapes_count.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            total_calls = sum(self._allgatherv_shapes_count.values())
            self._log("ALL_GATHERV:", use_print=use_print)
            self._log("  Total calls: %d", total_calls, use_print=use_print)
            self._log(
                "  Unique tensor shapes: %d",
                len(self._allgatherv_shapes_count),
                use_print=use_print,
            )
            for shape, count in sorted_shapes[:5]:  # Top 5
                percentage = (count / total_calls) * 100
                numel = np.prod(shape) if shape else 0
                self._log(
                    "    Shape %s: %d calls (%.2f%%), %s elements",
                    shape,
                    count,
                    percentage,
                    format(numel, ","),
                    use_print=use_print,
                )
            self._log("-" * 80, use_print=use_print)

        # Print reduce_scatterv statistics
        if (
            hasattr(self, "_reducescatterv_shapes_count")
            and self._reducescatterv_shapes_count
        ):
            sorted_shapes = sorted(
                self._reducescatterv_shapes_count.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            total_calls = sum(self._reducescatterv_shapes_count.values())
            self._log("REDUCE_SCATTERV:", use_print=use_print)
            self._log("  Total calls: %d", total_calls, use_print=use_print)
            self._log(
                "  Unique tensor shapes: %d",
                len(self._reducescatterv_shapes_count),
                use_print=use_print,
            )
            for shape, count in sorted_shapes[:5]:  # Top 5
                percentage = (count / total_calls) * 100
                numel = np.prod(shape) if shape else 0
                self._log(
                    "    Shape %s: %d calls (%.2f%%), %s elements",
                    shape,
                    count,
                    percentage,
                    format(numel, ","),
                    use_print=use_print,
                )
            self._log("-" * 80, use_print=use_print)

        self._log("=" * 80, use_print=use_print)

    def __del__(self):
        try:
            # Cleanup Perun communicator
            # Note: Statistics are printed by atexit handler, not here
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
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")

        # Handle list of tensors
        is_list_input = isinstance(input_, list)
        if is_list_input:
            results = []
            for tensor in input_:
                result = self._all_gatherv_single(tensor, dim, sizes)
                results.append(result)
            return results
        else:
            return self._all_gatherv_single(input_, dim, sizes)

    def _all_gatherv_single(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if self.disabled:
            return input_
        # Stats (best-effort)
        shape_key = tuple(input_.shape)
        self._allgatherv_shapes_count[shape_key] += 1

        input_size = input_.size()
        if sizes is not None:
            assert len(sizes) == self.world_size
            assert input_.shape[dim] == sizes[self.rank], (
                f"{input_.shape[dim]} != {sizes[self.rank]}"
            )
            output_size = (sum(sizes),) + input_size[1:]
        else:
            output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # Perun requires contiguous flat tensors and byte-level counts/displacements
        input_contiguous = input_.contiguous()
        input_flat = input_contiguous.view(-1)
        output_flat = output_tensor.view(-1)

        # Calculate byte counts and displacements for each rank
        itemsize = input_flat.element_size()
        elems_per_rank = (
            [s * (input_flat.numel() // input_.shape[0]) for s in sizes]
            if sizes
            else [input_flat.numel()] * self.world_size
        )
        recvcounts_bytes = [s * itemsize for s in elems_per_rank]

        displs_bytes = []
        offset = 0
        for count in recvcounts_bytes:
            displs_bytes.append(offset)
            offset += count

        # Convert to numpy arrays for ctypes
        rc = np.asarray(recvcounts_bytes, dtype=np.uint32)
        dp = np.asarray(displs_bytes, dtype=np.uint32)

        # Call Perun allgatherv
        stream = self._get_current_stream()
        perun.allgatherv(
            self._comm,
            stream.cuda_stream,
            input_flat.data_ptr(),
            input_flat.numel() * itemsize,
            output_flat.data_ptr(),
            rc.ctypes.data,
            dp.ctypes.data,
        )

        return output_tensor

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        assert input_tensor.device == self._device, (
            f"this perun communicator is created to work on {self._device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        # Stats (best-effort)
        shape_key = tuple(input_tensor.shape)
        self._reducescatterv_shapes_count[shape_key] += 1

        # Require caller-provided tensors to be contiguous
        assert input_tensor.is_contiguous(), "input_tensor must be contiguous"
        assert output_tensor.is_contiguous(), "output_tensor must be contiguous"

        # Convert sizes (rows per rank) to element counts per rank
        assert len(sizes) == self.world_size
        assert input_tensor.dim() >= 1
        elems_per_row = input_tensor.numel() // input_tensor.shape[0]
        recvcounts_elems = [int(s) * elems_per_row for s in sizes]
        rc = np.asarray(recvcounts_elems, dtype=np.uint32)

        comm_val = self._comm
        # stream_ptr = self._get_current_stream().cuda_stream
        stream_ptr = stream.cuda_stream
        send_ptr = input_tensor.data_ptr()
        sendcount_val = input_tensor.numel()
        recv_ptr = output_tensor.data_ptr()
        recvcounts_ptr = rc.ctypes.data

        perun.reducescatterv(
            comm_val,
            stream_ptr,
            send_ptr,
            sendcount_val,
            recv_ptr,
            recvcounts_ptr,
            input_tensor.dtype,
            op,
        )

    def all_reduce(self, input_):
        # Collect tensor shape statistics
        # shape_key = tuple(input_.shape)
        # self._allreduce_shapes_count[shape_key] += 1

        # output = self._get_output_buffer(input_)
        output = torch.empty_like(input_)
        perun.allreduce(
            self._comm,
            self._get_current_stream().cuda_stream,
            input_.data_ptr(),
            input_.numel(),
            output.data_ptr(),
            input_.dtype,
            torch.distributed.ReduceOp.SUM,
        )
        return output
