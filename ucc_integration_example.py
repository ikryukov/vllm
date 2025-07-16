#!/usr/bin/env python3
"""
Example showing how to integrate UCCAllreduce into distributed communication
similar to how it would be used in vllm.
"""

import os
import torch
import torch.distributed as dist
from ucc_allreduce_standalone import UCCAllreduce


def setup_distributed():
    """Setup distributed environment for UCC."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="ucc",
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0))
        )
    return dist.group.WORLD


def simulate_model_communication():
    """Simulate how UCCAllreduce would be used in model communication."""

    # Setup
    group = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Setting up UCCAllreduce...")

    # Create UCC allreduce instance
    ucc_allreduce = UCCAllreduce(group, device)

    # Simulate model gradients of different sizes
    gradient_sizes = [
        (1024, 1024),      # Large gradient (4MB)
        (512, 512),        # Medium gradient (1MB)
        (256, 256),        # Small gradient (256KB)
        (128, 128),        # Tiny gradient (64KB)
    ]

    for i, (h, w) in enumerate(gradient_sizes):
        print(f"\n[Rank {rank}] Testing gradient {i+1}: {h}x{w}")

        # Create gradient tensor with rank-specific values
        gradient = torch.randn(h, w, device=device) * (rank + 1)
        original_sum = gradient.sum().item()

        print(f"[Rank {rank}] Original gradient sum: {original_sum:.2f}")

        # Check if we should use UCC allreduce
        should_use_ucc = ucc_allreduce.should_use_ucc_allreduce(gradient)
        print(f"[Rank {rank}] Should use UCC: {should_use_ucc}")

        # Perform allreduce
        result = ucc_allreduce.custom_allreduce(gradient.clone())

        if result is not None:
            result_sum = result.sum().item()
            print(f"[Rank {rank}] UCC allreduce result sum: {result_sum:.2f}")

            # Verify the result (should be sum of all ranks)
            expected_sum = sum(original_sum * (r + 1) / (rank + 1) for r in range(world_size))
            print(f"[Rank {rank}] Expected sum: {expected_sum:.2f}")

        else:
            print(f"[Rank {rank}] UCC allreduce returned None, using fallback")
            # In real usage, would fall back to regular allreduce
            dist.all_reduce(gradient)
            fallback_sum = gradient.sum().item()
            print(f"[Rank {rank}] Fallback allreduce sum: {fallback_sum:.2f}")


def benchmark_communication():
    """Simple benchmark comparing UCC vs regular allreduce."""
    import time

    group = setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank()

    ucc_allreduce = UCCAllreduce(group, device)

    # Test tensor
    test_tensor = torch.randn(2048, 2048, device=device)

    print(f"\n[Rank {rank}] Benchmarking communication...")

    # Warm up
    for _ in range(3):
        ucc_allreduce.custom_allreduce(test_tensor.clone())
        dist.all_reduce(test_tensor.clone())

    # Benchmark UCC allreduce
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        result = ucc_allreduce.custom_allreduce(test_tensor.clone())
    torch.cuda.synchronize()
    ucc_time = time.time() - start_time

    # Benchmark regular allreduce
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        dist.all_reduce(test_tensor.clone())
    torch.cuda.synchronize()
    regular_time = time.time() - start_time

    print(f"[Rank {rank}] UCC allreduce time: {ucc_time:.4f}s")
    print(f"[Rank {rank}] Regular allreduce time: {regular_time:.4f}s")

    if ucc_time > 0:
        speedup = regular_time / ucc_time
        print(f"[Rank {rank}] UCC speedup: {speedup:.2f}x")


if __name__ == "__main__":
    try:
        print("=== UCCAllreduce Integration Example ===")

        # Test basic model communication patterns
        simulate_model_communication()

        # Simple benchmark
        if torch.cuda.is_available():
            benchmark_communication()

        print(f"\n[Rank {dist.get_rank()}] ✓ Integration example completed successfully!")

    except Exception as e:
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 'N/A'}] ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()