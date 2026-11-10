import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.bench_kernels import run_benchmarks


def test_mega_kernels_benchmark_runs_cleanly():
    # Verify that the entire hardware roofline benchmark suite completes without error
    run_benchmarks()
