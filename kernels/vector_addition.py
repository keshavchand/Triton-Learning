from math import ceil

import triton
import triton.language as tl
import torch

@triton.jit
def _vectorAdd(
  In1, In2, Out,
  ElementCount,
  blockSize: tl.constexpr,
):
  pid = tl.program_id(0)
  idx = (pid * blockSize + tl.arange(0, blockSize))
  mask = idx < ElementCount
  
  In1Data = tl.load(In1 + idx, mask = mask, other = 0)
  In2Data = tl.load(In2 + idx, mask = mask, other = 0)
  result = In1Data + In2Data
  tl.store(Out + idx, value=result, mask = mask)
pass

def vectorAdd(A: torch.Tensor, B: torch.Tensor, blockSize = (16,)) -> torch.Tensor:
  assert A.shape[-1] == B.shape[-1], "The tensors must be of same shape"
  assert len(A.shape) == 1, "Currently only 1D is supported"
  assert blockSize[-1] >= 16, "block size must be greater than 16"

  C = torch.empty((A.shape[-1]), dtype=A.dtype, device=A.device)
  gridSize = (ceil(C.shape[-1] / blockSize[-1]), 1)
  _vectorAdd[gridSize](
    A, B, C,
    A.shape[-1],
    blockSize[-1],
  )

  return C
