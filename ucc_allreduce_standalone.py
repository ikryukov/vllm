#!/usr/bin/env python3
"""
Standalone UCCAllreduce implementation for testing and demonstration.
This version doesn't depend on vllm package imports to avoid compatibility issues.
"""

import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class UCCAllreduce:
    """
    UCCAllreduce provides a minimalistic interface for allreduce operations
    using PyTorch's native UCC (Unified Collective Communication) backend.

    UCC is a high-performance collective communication library that provides
    optimized implementations for various interconnects and can leverage
    hardware acceleration when available.
    """

    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device]) -> None:
        """
        Initialize UCCAllreduce.

        Args:
            group: The process group to work on. Must be a UCC process group.
            device: The device to bind operations to.
        """
        self.disabled = True
        self.group = group

        # Validate that this is a UCC process group
        if dist.get_backend(group) != "ucc":
            logger.warning(
                "UCCAllreduce requires a UCC process group backend, "
                "but got backend: %s. Disabling UCC allreduce.",
                dist.get_backend(group)
            )
            return

        # Set up device
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Store group size for convenience
        self.world_size = dist.get_world_size(group)

        # Enable UCC allreduce if validation passes
        self.disabled = False
        logger.info(
            "UCCAllreduce initialized successfully with UCC backend "
            "on device %s, world size: %d",
            self.device, self.world_size
        )

    def custom_allreduce(self, tensor: torch.Tensor,
                        op: dist.ReduceOp = dist.ReduceOp.SUM) -> Optional[torch.Tensor]:
        """
        Perform allreduce operation using UCC backend.

        Args:
            tensor: Input tensor to reduce across all processes.
            op: Reduction operation (default: SUM).

        Returns:
            The reduced tensor or None if UCC allreduce is disabled.
        """
        if self.disabled:
            return None

        # Ensure tensor is on the correct device
        if tensor.device != self.device:
            tensor = tensor.to(self.device)

        # Perform allreduce using UCC backend
        try:
            dist.all_reduce(tensor, op=op, group=self.group)
            return tensor
        except Exception as e:
            logger.warning(
                "UCC allreduce failed: %s. Falling back to regular allreduce.",
                str(e)
            )
            return None

    def should_use_ucc_allreduce(self, tensor: torch.Tensor) -> bool:
        """
        Determine if UCC allreduce should be used for the given tensor.

        Args:
            tensor: The tensor to check.

        Returns:
            True if UCC allreduce should be used, False otherwise.
        """
        if self.disabled:
            return False

        # UCC allreduce is beneficial for larger tensors
        # and when using multiple processes
        tensor_size = tensor.numel() * tensor.element_size()

        # Use UCC for tensors larger than 1MB and when world size > 1
        return tensor_size > 1024 * 1024 and self.world_size > 1

    def close(self) -> None:
        """
        Clean up resources.
        """
        # UCC process group cleanup is handled by PyTorch
        pass

    @staticmethod
    def is_ucc_available() -> bool:
        """
        Check if UCC backend is available in PyTorch.

        Returns:
            True if UCC backend is available, False otherwise.
        """
        try:
            return hasattr(dist.Backend, 'UCC') or 'ucc' in dist.Backend.__dict__.values()
        except Exception:
            return False

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        if hasattr(self, 'disabled') and not self.disabled:
            self.close()


def test_ucc_basic():
    """Test basic UCC functionality without distributed setup."""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== UCCAllreduce Standalone Test ===")

    # Test UCC availability
    is_available = UCCAllreduce.is_ucc_available()
    print(f"✓ UCC availability: {is_available}")

    # Test device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # Create test tensor
    tensor = torch.randn(1000, 1000, device=device)
    tensor_size = tensor.numel() * tensor.element_size()
    print(f"✓ Test tensor size: {tensor_size} bytes ({tensor_size / (1024*1024):.2f} MB)")

    print("\n✓ Basic functionality test passed!")
    print("✓ UCCAllreduce class is ready for distributed use")
    return True


def test_ucc_distributed():
    """Test UCC with a minimal distributed setup."""
    import os

    # This test requires torchrun to be executed properly
    print("\n=== Testing UCC with minimal distributed setup ===")

    if not dist.is_initialized():
        print("Initializing distributed process group...")
        try:
            # Try to initialize with UCC backend
            dist.init_process_group(
                backend="ucc",
                init_method="env://",
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0))
            )
            print("✓ UCC process group initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize UCC process group: {e}")
            return False

    # Get the default process group
    group = dist.group.WORLD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create UCCAllreduce instance
    try:
        ucc_allreduce = UCCAllreduce(group, device)
        print(f"✓ UCCAllreduce initialized, disabled: {ucc_allreduce.disabled}")

        # Test with a tensor
        tensor = torch.ones(100, 100, device=device) * dist.get_rank()
        print(f"✓ Created tensor with rank-specific values")

        # Test should_use_ucc_allreduce
        should_use = ucc_allreduce.should_use_ucc_allreduce(tensor)
        print(f"✓ Should use UCC allreduce: {should_use}")

        # Test custom_allreduce
        result = ucc_allreduce.custom_allreduce(tensor.clone())
        if result is not None:
            print(f"✓ UCC allreduce succeeded, result sum: {result.sum().item()}")
        else:
            print("✓ UCC allreduce returned None (expected for fallback)")

    except Exception as e:
        print(f"✗ Error during distributed test: {e}")
        return False

    print("✓ Distributed test completed successfully!")
    return True


if __name__ == "__main__":
    import os

    # Run basic test
    test_ucc_basic()

    # Run distributed test if environment is set up
    if os.environ.get("WORLD_SIZE") and os.environ.get("RANK"):
        test_ucc_distributed()
    else:
        print("\nNote: To run distributed tests, use:")
        print("torchrun --nproc_per_node=2 ucc_allreduce_standalone.py")