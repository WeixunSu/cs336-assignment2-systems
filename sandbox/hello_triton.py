import os

# 必须在 import triton 之前设置，强制加载 CPU InterpreterDriver
os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl

@triton.jit
def kernel_add(in1_ptr, in2_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # get pid
    pid = tl.program_id(axis=0)

    # where this thread starts
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # edge handling
    mask = offsets < n_elements

    # load data
    x = tl.load(in1_ptr + offsets, mask=mask)
    y = tl.load(in2_ptr + offsets, mask=mask)

    # addition
    z = x + y

    # write 
    tl.store(out_ptr + offsets, z, mask=mask)

    return

def triton_add(x: torch.Tensor, y: torch.Tensor)-> torch.Tensor:
    # alloc output memory
    out = torch.empty_like(x)

    # boundary
    n_elements = out.numel()

    # grid size
    grid = lambda meta: (tl.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # kernel launch
    kernel_add[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


if __name__ == "__main__":
    print("Hello Triton")
    torch.manual_seed(0)

    shape = (1027,)

    x = torch.randn(shape)
    y = torch.randn(shape)
    z = triton_add(x, y)

    # Check accuracy
    golden = torch.add(x, y)
    result = torch.allclose(z, golden)

    if result == True:
        print("PASS")
    else:
        print("FAIL")