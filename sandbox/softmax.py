
import os

os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

# simple kernel, last dim reduction, fit in one block
@triton.jit
def softmax_kernel_last_dim_simple(in_ptr, out_ptr, n_cols, BLOCK_SIZE_COL: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * n_cols + tl.arange(0, BLOCK_SIZE_COL)
    mask = offsets < ((pid + 1) * n_cols)

    # load data
    x = tl.load(in_ptr + offsets, mask=mask, other=-float("inf"))
    # softmax
    row_max = tl.max(x)
    row_sub = x - row_max
    row_exp = tl.exp(row_sub)
    row_sum = tl.sum(row_exp)
    out = row_exp / row_sum

    tl.store(out_ptr + offsets, out, mask=mask)
    return

# 2D kernel, col size fits in BLOCK_SIZE, for small columns
# @triton.jit
# def softmax_kernel_last_dim(in_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
#     # pid
#     pid_r = tl.program_id(axis=0) # row
#     pid_c = tl.program_id(axis=1) # col

#     # assert doesn't work inside kernel
#     # assert(n_cols <= BLOCK_SIZE_COL)

#     # block
    

#     # edge
    

#     return

def triton_softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    out = torch.empty_like(x)

    assert(out.ndim <= 2)

    n_cols = out.shape[-1]
    n_rows = 1 if out.ndim == 1 else out.shape[0]

    grid = lambda meta: (n_rows,)

    softmax_kernel_last_dim_simple[grid](x, out, n_cols, BLOCK_SIZE_COL=1024)

    return out

def compare_result(out_test, out_golden):
    result = torch.allclose(out_test, out_golden)
    if result == True:
        print("PASS")
    else:
        print("Test: ", out_test)
        print("Golden: ", out_golden)
        print("Mismatch!")
        print("abs distance: ", out_golden - out_test)
    return result

def run_case(case_str, s):
    print("Running Case: ", case_str)
    x = torch.randn(s)
    # print("host x: ", x)
    out_test = triton_softmax(x)
    out_golden = torch.softmax(x, -1)
    return compare_result(out_test, out_golden)
    

if __name__ == "__main__":
    torch.manual_seed(0)
    run_case("0", (8,))
    run_case("1", (1, 8))
    run_case("2", (4, 8))
    run_case("3", (1, 512))
    run_case("4", (8, 512))
    run_case("5", (8, 888))

    