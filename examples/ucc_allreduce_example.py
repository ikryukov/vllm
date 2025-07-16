#!/usr/bin/env python3
"""
Example demonstrating UCCAllreduce usage with PyTorch's UCC backend.

This example shows how to:
1. Initialize a UCC process group
2. Create a UCCAllreduce instance
3. Perform allreduce operations

Requirements:
- PyTorch with UCC backend support
- UCC library installed
- Multiple processes (run with: torchrun --nproc_per_node=2 ucc_allreduce_example.py)
"""

import os
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.ucc_allreduce import UCCAllreduce


def main():
    # Initialize the process group with UCC backend
    # This requires PyTorch to be built with UCC support
    dist.init_process_group(
        backend="ucc",
        init_method="env://",
        world_size=int(os.environ.get("WORLD_SIZE", 2)),
        rank=int(os.environ.get("RANK", 0))
    )

    print(f"Process {dist.get_rank()} of {dist.get_world_size()} initialized")

    # Create a UCC process group (using the default group for simplicity)
    ucc_group = dist.distributed_c10d._get_default_group()

    # Initialize UCCAllreduce
    device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
    ucc_allreduce = UCCAllreduce(group=ucc_group, device=device)

    if ucc_allreduce.disabled:
        print(f"UCCAllreduce is disabled on rank {dist.get_rank()}")
        return

    # Create a test tensor
    tensor = torch.ones(4, device=device) * (dist.get_rank() + 1)
    print(f"Rank {dist.get_rank()} initial tensor: {tensor}")

    # Check if we should use UCC allreduce
    if ucc_allreduce.should_use_ucc_allreduce(tensor):
        print(f"Rank {dist.get_rank()} using UCC allreduce")

        # Perform allreduce using the custom_allreduce method
        result = ucc_allreduce.custom_allreduce(tensor)
        if result is not None:
            print(f"Rank {dist.get_rank()} after UCC allreduce: {result}")
        else:
            print(f"Rank {dist.get_rank()} UCC allreduce returned None")
    else:
        print(f"Rank {dist.get_rank()} not using UCC allreduce")

        # Fallback to regular PyTorch allreduce
        dist.all_reduce(tensor, group=ucc_group)
        print(f"Rank {dist.get_rank()} after regular allreduce: {tensor}")

    # Clean up
    ucc_allreduce.close()
    dist.destroy_process_group()
    print(f"Rank {dist.get_rank()} finished")


if __name__ == "__main__":
    main()