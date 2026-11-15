
import os

import torch
import triton
import triton.language as tl

# 1D simple kernel
@triton.jit
def softmax_kernel_last_dim_1d(in_ptr, out_ptr, n_cols, BLOCK_SIZE_COL: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * n_cols

    return

# 2D kernel, col size fits in BLOCK_SIZE, for small columns
@triton.jit
def softmax_kernel_last_dim(in_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    # pid
    pid_r = tl.program_id(axis=0) # row
    pid_c = tl.program_id(axis=1) # col

    # assert doesn't work inside kernel
    # assert(n_cols <= BLOCK_SIZE_COL)

    # block
    

    # edge
    

    return

def triton_softmax(x: torch.Tensor, dim=-1) -> torch.Tensor:
    out = torch.empty_like(x)

    assert(out.ndim <= 2)

    n_cols = out.shape[-1]
    n_rows = 1 if out.ndim == 1 else out.shape[0]

    grid = lambda meta: (tl.cdiv(n_rows, meta["BLOCK_SIZE_ROW"], tl.cdiv(n_cols, meta["BLOCK_SIZE_COL"]),))

    softmax_kernel_last_dim[grid](x, out, n_rows, n_cols, BLOCK_SIZE_ROW=1, BLOCK_SIZE_COL=1024)

    return out

if __name__ == "__main__"
    torch.manual_seed(0)

    x = torch.randn(8, 1024)

    out_test = triton_softmax(x)
    out_golden = torch.softmax(x)

    compare_result = torch.allclose(out_test, out_golden)
    if compare_result == True:
        print("PASS")
    else:
        print("FAIL")