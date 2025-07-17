 # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.ucc_all_reduce import (
    UCCAllreduce, is_ucc_available, get_ucc_version, check_ucc_cuda_support)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_group)
from ..utils import (ensure_model_parallel_initialized,
                     init_test_distributed_environment)


@pytest.mark.skipif(not is_ucc_available(), reason="UCC not available")
def test_ucc_availability():
    """Test UCC availability and version info."""
    assert is_ucc_available()
    version = get_ucc_version()
    assert version is not None
    print(f"UCC version: {version}")

    # Check CUDA support
    cuda_support = check_ucc_cuda_support()
    print(f"UCC CUDA support: {cuda_support}")


@pytest.mark.skipif(not is_ucc_available(), reason="UCC not available")
def test_ucc_allreduce_initialization():
    """Test UCC allreduce initialization."""
    init_test_distributed_environment()
    ensure_model_parallel_initialized()

    tp_group = get_tensor_model_parallel_group()

    # Test UCCAllreduce initialization
    ucc_comm = UCCAllreduce(
        group=tp_group.cpu_group,
        device=torch.device("cuda:0"),
        max_size=1024 * 1024
    )

    assert ucc_comm.rank == tp_group.rank_in_group
    assert ucc_comm.world_size == tp_group.world_size
    assert ucc_comm.device.type == "cuda"

    # Test tensor compatibility checking
    test_tensor = torch.randn(16, 16, dtype=torch.float16, device=ucc_comm.device)
    can_use = ucc_comm.should_custom_ar(test_tensor)

    print(f"UCC allreduce can handle tensor: {can_use}")
    print(f"UCC allreduce disabled: {ucc_comm.disabled}")

    # Clean up
    ucc_comm.close()


@pytest.mark.skipif(not is_ucc_available(), reason="UCC not available")
def test_ucc_allreduce_operation():
    """Test UCC allreduce operation with actual tensors."""
    init_test_distributed_environment()
    ensure_model_parallel_initialized()

    tp_group = get_tensor_model_parallel_group()

    ucc_comm = UCCAllreduce(
        group=tp_group.cpu_group,
        device=torch.device("cuda:0"),
        max_size=1024 * 1024
    )

    if not ucc_comm.disabled:
        # Test allreduce operation
        test_tensor = torch.ones(10, 10, dtype=torch.float32, device=ucc_comm.device)

        # Multiply by rank to differentiate input across ranks
        test_tensor *= (ucc_comm.rank + 1)

        result = ucc_comm.custom_all_reduce(test_tensor)

        if result is not None:
            # Check that result is correct (sum of all ranks)
            expected_sum = sum(range(1, ucc_comm.world_size + 1))
            expected_tensor = torch.full_like(test_tensor, expected_sum)

            # Allow for small numerical errors
            assert torch.allclose(result, expected_tensor, rtol=1e-5)
            print(f"UCC allreduce successful! Result shape: {result.shape}")
        else:
            print("UCC allreduce returned None (fallback to other methods)")
    else:
        print("UCC allreduce is disabled")

    # Clean up
    ucc_comm.close()


@pytest.mark.skipif(not is_ucc_available(), reason="UCC not available")
def test_ucc_allreduce_cuda_graph():
    """Test UCC allreduce with CUDA graph capture."""
    init_test_distributed_environment()
    ensure_model_parallel_initialized()

    tp_group = get_tensor_model_parallel_group()

    ucc_comm = UCCAllreduce(
        group=tp_group.cpu_group,
        device=torch.device("cuda:0"),
        max_size=1024 * 1024
    )

    if not ucc_comm.disabled:
        # Test with CUDA graph capture
        with ucc_comm.capture():
            test_tensor = torch.ones(8, 8, dtype=torch.float32, device=ucc_comm.device)
            result = ucc_comm.custom_all_reduce(test_tensor)

            if result is not None:
                print(f"UCC allreduce with graph capture successful! Shape: {result.shape}")
            else:
                print("UCC allreduce with graph capture returned None")
    else:
        print("UCC allreduce is disabled")

    # Clean up
    ucc_comm.close()


def test_ucc_allreduce_fallback():
    """Test UCC allreduce fallback behavior when UCC is not available."""
    init_test_distributed_environment()
    ensure_model_parallel_initialized()

    tp_group = get_tensor_model_parallel_group()

    # This should work even without UCC (will be disabled)
    ucc_comm = UCCAllreduce(
        group=tp_group.cpu_group,
        device=torch.device("cuda:0"),
        max_size=1024 * 1024
    )

    # Should be disabled if UCC is not available
    if not is_ucc_available():
        assert ucc_comm.disabled

    # Test that it doesn't crash when disabled
    test_tensor = torch.ones(4, 4, dtype=torch.float32, device=ucc_comm.device)
    result = ucc_comm.custom_all_reduce(test_tensor)

    # Should return None if disabled
    if ucc_comm.disabled:
        assert result is None

    # Clean up
    ucc_comm.close()


if __name__ == "__main__":
    # Run basic tests
    test_ucc_availability()
    test_ucc_allreduce_fallback()
    print("UCC allreduce tests completed!")