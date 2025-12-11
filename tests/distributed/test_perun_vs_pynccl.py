# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os

import pytest
import torch

from vllm.distributed.device_communicators.perun_communicator import (
    PerunCommunicator,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import (
    get_world_group,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables


def distributed_run(
    fn,
    world_size,
    extra_env: dict[str, str] | None = None,
    fn_args: tuple = (),
    fn_kwargs: dict | None = None,
):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12346"

        if extra_env:
            env.update(extra_env)
        p = multiprocessing.Process(
            target=fn,
            args=(env, *fn_args),
            kwargs={} if fn_kwargs is None else fn_kwargs,
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    def wrapped_fn(env, *args, **kwargs):
        update_environment_variables(env)
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        fn(*args, **kwargs)

    return wrapped_fn


@worker_fn_wrapper
def reduce_scatterv_comparison_worker_fn():
    """Compare reduce_scatterv between Perun and PyNccl communicators."""
    world_group = get_world_group()
    world_size = world_group.world_size
    rank = world_group.rank

    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank} device: {device}")
    torch.cuda.set_device(device)
    # torch.cuda.set_sync_debug_mode(2)

    # Initialize both communicators
    pynccl_comm = PyNcclCommunicator(world_group.cpu_group, device=device)
    perun_comm = PerunCommunicator(world_group.cpu_group, device=device)

    if perun_comm.disabled:
        pytest.skip("Perun communicator is disabled")

    # Test with simpler 2D tensor first for easier debugging
    # Each rank creates: [[1*rank_id, 1*rank_id], [2*rank_id, 2*rank_id], ...]
    sizes = [4, 4][:world_size]  # Each rank gets 4 rows
    total_size = sum(sizes)
    hidden_dim = 4  # Small for easy debugging

    # Create simple input tensor for verification
    # Rank 0: [[0, 0, 0, 0], [0, 0, 0, 0], ..., [0, 0, 0, 0]]  (8 rows)
    # Rank 1: [[1, 1, 1, 1], [1, 1, 1, 1], ..., [1, 1, 1, 1]]  (8 rows)
    input_tensor = (
        torch.ones(total_size, hidden_dim, dtype=torch.float32, device=device) * rank
    )
    input_perun = (
        torch.ones(total_size, hidden_dim, dtype=torch.float32, device=device) * rank
    )

    # Add row indices to make it easier to see which rows went where
    for i in range(total_size):
        input_tensor[i] *= i + 1
        input_perun[i] *= i + 1

    print(f"Rank {rank} input (first 8 rows):\n{input_tensor[:8]}")

    # PyNccl reduce_scatterv
    input_pynccl = input_tensor.clone()
    output_pynccl = torch.zeros(
        sizes[rank], hidden_dim, dtype=torch.float32, device=device
    )
    pynccl_comm.reduce_scatterv(output_pynccl, input_pynccl, sizes=sizes)
    torch.cuda.synchronize(device)

    # Perun reduce_scatterv
    print(f"Rank {rank} input_perun (first 8 rows):\n{input_perun[:8]}")
    output_perun = torch.zeros(
        sizes[rank], hidden_dim, dtype=torch.float32, device=device
    )
    perun_comm.reduce_scatterv(output_perun, input_perun, sizes=sizes)
    torch.cuda.synchronize(device)

    print(f"Rank {rank} PyNccl output:\n{output_pynccl}")
    print(f"Rank {rank} Perun output:\n{output_perun}")

    # Compare results
    torch.testing.assert_close(
        output_pynccl,
        output_perun,
        msg=f"Rank {rank}: Perun and PyNccl reduce_scatterv results differ",
    )

    print(f"Rank {rank}: reduce_scatterv test passed!")


@worker_fn_wrapper
def allreduce_comparison_worker_fn(dtype: torch.dtype):
    """Compare allreduce between Perun and PyNccl communicators."""
    world_group = get_world_group()
    world_size = world_group.world_size
    rank = world_group.rank

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Initialize both communicators
    pynccl_comm = PyNcclCommunicator(world_group.cpu_group, device=device)
    perun_comm = PerunCommunicator(world_group.cpu_group, device=device)

    if perun_comm.disabled:
        pytest.skip("Perun communicator is disabled")

    # Each rank i initializes the tensor to (i + 1)
    shape = (8, 1024)
    input_tensor = torch.full(shape, fill_value=rank + 1, dtype=dtype, device=device)
    input_perun = input_tensor.clone()

    # PyNccl allreduce
    out_pynccl = pynccl_comm.all_reduce(input_tensor)
    torch.cuda.synchronize(device)
    # world_group.barrier()

    # Perun allreduce
    out_perun = perun_comm.all_reduce(input_perun)
    torch.cuda.synchronize(device)
    # world_group.barrier()

    # Compare results between backends
    torch.testing.assert_close(
        out_pynccl,
        out_perun,
        msg=f"Rank {rank}: Perun and PyNccl allreduce results differ",
    )

    # Also validate against the expected sum across ranks
    expected_val = (world_size * (world_size + 1)) / 2.0
    expected = torch.full(shape, fill_value=expected_val, dtype=dtype, device=device)
    torch.testing.assert_close(
        out_pynccl,
        expected,
        msg=f"Rank {rank}: allreduce result mismatches expected sum",
    )

    print(f"Rank {rank}: allreduce test passed!")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to run the test."
)
def test_perun_vs_pynccl_reduce_scatterv():
    """Test reduce_scatterv: Perun vs PyNccl on 2 ranks with 2D tensor (8, 3072)."""
    distributed_run(reduce_scatterv_comparison_worker_fn, 2)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to run the test."
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"]
)
def test_perun_vs_pynccl_allreduce(dtype: torch.dtype):
    """Test allreduce: Perun vs PyNccl where rank i uses value (i+1)."""
    distributed_run(allreduce_comparison_worker_fn, 2, fn_args=(dtype,))


@worker_fn_wrapper
def allgatherv_comparison_worker_fn(dtype: torch.dtype):
    """Compare allgatherv between Perun and PyNccl communicators."""
    world_group = get_world_group()
    world_size = world_group.world_size
    rank = world_group.rank

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Initialize both communicators
    pynccl_comm = PyNcclCommunicator(world_group.cpu_group, device=device)
    perun_comm = PerunCommunicator(world_group.cpu_group, device=device)

    if perun_comm.disabled:
        pytest.skip("Perun communicator is disabled")

    # Each rank i initializes its local tensor to (i + 1)
    sizes = [4, 4][:world_size]
    hidden_dim = 8
    local_rows = sizes[rank]
    input_tensor = torch.full(
        (local_rows, hidden_dim), fill_value=rank + 1, dtype=dtype, device=device
    )

    # PyNccl all_gatherv
    output_pynccl = torch.empty((sum(sizes), hidden_dim), dtype=dtype, device=device)
    pynccl_comm.all_gatherv(output_pynccl, input_tensor, sizes=sizes)
    torch.cuda.synchronize(device)

    # Perun all_gatherv
    output_perun = perun_comm.all_gatherv(input_tensor, dim=0, sizes=sizes)
    torch.cuda.synchronize(device)

    print(f"Rank {rank} PyNccl output:\n{output_pynccl}")
    print(f"Rank {rank} Perun output:\n{output_perun}")

    # Compare results between backends
    torch.testing.assert_close(
        output_pynccl,
        output_perun,
        msg=f"Rank {rank}: Perun and PyNccl allgatherv results differ",
    )

    # Validate expected concatenation order [rank0, rank1, ...]
    expected_chunks = [
        torch.full((sizes[r], hidden_dim), fill_value=r + 1, dtype=dtype, device=device)
        for r in range(world_size)
    ]
    expected = torch.cat(expected_chunks, dim=0)
    torch.testing.assert_close(
        output_pynccl,
        expected,
        msg=f"Rank {rank}: allgatherv result mismatches expected concatenation",
    )

    print(f"Rank {rank}: allgatherv test passed!")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Need at least 2 GPUs to run the test."
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"]
)
def test_perun_vs_pynccl_allgatherv(dtype: torch.dtype):
    """Test allgatherv: Perun vs PyNccl where rank i uses value (i+1)."""
    distributed_run(allgatherv_comparison_worker_fn, 2, fn_args=(dtype,))


if __name__ == "__main__":
    # For direct execution
    test_perun_vs_pynccl_reduce_scatterv()
