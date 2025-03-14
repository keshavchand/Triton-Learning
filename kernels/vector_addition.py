from math import ceil

import triton
import triton.language as tl
import torch

@triton.autotune(
  configs = [
    triton.Config({'blockSize': 128}, num_warps=4),
    triton.Config({'blockSize': 256}, num_warps=4),
    triton.Config({'blockSize': 512}, num_warps=4),
  ], key = ['ElementCount'], 
)
@triton.jit
def _vectorAdd_(
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

@triton.autotune(
  configs = [
    triton.Config({'blockSize': 128}, num_warps=4),
    triton.Config({'blockSize': 256}, num_warps=4),
    triton.Config({'blockSize': 512}, num_warps=4),
  ], key = [ 
    'rows', 'cols',
  ], 
)
@triton.jit
def _vectorAdd2d_(
  In1, In2, Out,
  rows, cols, 
  In1RowStride, In1ColStride,
  In2RowStride, In2ColStride,
  OutRowStride, OutColStride,

  blockSize: tl.constexpr,
):
  row = blockSize * tl.program_id(0)
  col = blockSize * tl.program_id(1)

  rows = (row + tl.arange(0, blockSize))[:, None]
  cols = (col + tl.arange(0, blockSize))[None, :]
  mask = (rows < rows) & (cols < cols)

  in1Data = tl.load(In1 + (rows * In1RowStride) + (cols * In1ColStride), mask = mask, other = 0)
  in2Data = tl.load(In2 + (rows * In2RowStride) + (cols * In2ColStride), mask = mask, other = 0)

  result = in1Data + in2Data
  tl.store(Out + (rows * OutRowStride) + (cols * OutColStride), value = result, mask = mask)
pass

def vectorAdd(A: torch.Tensor, B: torch.Tensor, blockSize = (16,)) -> torch.Tensor:
  assert A.shape[-1] == B.shape[-1], "The tensors must be of same shape"
  assert len(A.shape) == 1, "Currently only 1D is supported"
  assert blockSize[-1] >= 16, "block size must be greater than 16"

  C = torch.empty((A.shape[-1]), dtype=A.dtype, device=A.device)
  gridSize = lambda meta : (triton.cdiv(C.shape[-1], meta['blockSize']), )
  _vectorAdd_[gridSize](
    A, B, C,
    A.shape[-1],
  )

  return C

def vectorAdd2D(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
  assert A.shape[-1] == B.shape[-1], f"Tensor must be of same shape {A.shape} vs {B.shape}"
  assert A.shape[-2] == B.shape[-2], f"Tensor must be of same shape {A.shape} vs {B.shape}"
  assert A.device == B.device, f"Tensor devices should be same {A.device} vs {B.device}"
  assert len(A.shape) == 2, "Currently only 2D is supported"

  C = torch.empty((A.shape[-2], A.shape[-1]), device=A.device)
  gridSize = lambda meta: (
    triton.cdiv(C.shape[-2], meta["blockSize"]),
    triton.cdiv(C.shape[-1], meta["blockSize"]),
  )

  ARowStride, AColStride = A.stride()[:2]
  BRowStride, BColStride = B.stride()[:2]
  OutRowStride, OutColStride = C.stride()[:2]

  _vectorAdd2d_[gridSize](
    A, B, C,
    A.shape[-2], A.shape[-1],
    ARowStride, AColStride,
    BRowStride, BColStride,
    OutRowStride, OutColStride,
  )

  return C
