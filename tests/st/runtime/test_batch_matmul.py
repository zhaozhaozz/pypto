# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
System tests for batch matrix multiplication operation.

This test validates the tensor.batch_matmul operation through the complete
compilation and execution pipeline, comparing results against PyTorch reference.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


class TestBatchMatmul(PTOTestCase):
    __test__ = False  # Not a pytest test class

    def __init__(self, batch: int = 2, m: int = 64, k: int = 64, n: int = 64, config=None):
        super().__init__(config)
        self.batch = batch
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"batch_matmul_{self.batch}x{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.batch, self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.batch, self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [self.batch, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, M, K, N = self.batch, self.M, self.K, self.N

        @pl.program
        class BatchMatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def batch_matmul(
                self,
                a: pl.Tensor[[B, M, K], pl.FP32],
                b: pl.Tensor[[B, K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                result = pl.tensor.batch_matmul(a, b)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[B, M, K], pl.FP32],
                b: pl.Tensor[[B, K, N], pl.FP32],
                c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
            ) -> pl.Tensor[[B, M, N], pl.FP32]:
                out_c = self.batch_matmul(a, b, c)
                return out_c

        return BatchMatmulProgram

    def compute_expected(self, tensors, params=None):
        """Compute reference output using PyTorch."""
        a = tensors["a"]
        b = tensors["b"]
        tensors["c"][:] = torch.bmm(a, b)


class TestBatchMatmulPTO(TestBatchMatmul):
    """Test batch_matmul with PTO backend and PTOAS optimization."""

    __test__ = False

    def get_name(self) -> str:
        return f"batch_matmul_pto_{self.batch}x{self.M}x{self.K}x{self.N}"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B


@pytest.mark.parametrize(
    "batch,m,k,n",
    [
        (2, 64, 64, 64),
        (4, 32, 32, 32),
        (1, 128, 64, 128),
    ],
)
def test_batch_matmul(test_runner, batch, m, k, n):
    """Test batch_matmul with various batch sizes and matrix shapes."""
    test_case = TestBatchMatmul(batch=batch, m=m, k=k, n=n)
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


@pytest.mark.parametrize(
    "batch,m,k,n",
    [
        (2, 64, 64, 64),
        (4, 32, 32, 32),
        (1, 128, 64, 128),
    ],
)
def test_batch_matmul_pto(test_runner, batch, m, k, n):
    """Test batch_matmul with PTO backend and PTOAS optimization."""
    test_case = TestBatchMatmulPTO(batch=batch, m=m, k=k, n=n)
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed (PTO): {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--forked"])
