from math import ceil
from typing import List

import triton
import torch
import triton.language as tl

def genConfigs(blockSizeRows: List[int], colSize: List[int], num_warps: List[int]):
  for br in blockSizeRows:
      for cs in colSize:
        for nw in num_warps:
          yield triton.Config({'blockSizeRows': br, 'blockSizeCols': br, 'colSize': cs}, num_warps = nw)

configs = [config for config in genConfigs(
        [128, 256],
        [128, 256],
        [4, 8],
      )]

import pdb

# @triton.autotune(
#   configs = configs, key = [
#     'In1Rows', 'In1Cols',
#     'In2Rows', 'In2Cols',
#     'OutRows', 'OutCols',
#   ]
# )
@triton.jit
def _matmul_(
  In1, In2, Out,
  In1Rows, In1Cols,
  In2Rows, In2Cols,
  OutRows, OutCols,

  blockSizeRows: tl.constexpr,
  blockSizeCols: tl.constexpr,
  colSize: tl.constexpr,
):
  row = tl.program_id(0) * blockSizeRows
  col = tl.program_id(1) * blockSizeCols

  """
    Alog: 
      for i in range(..)
        a = A[row * blockSizeRows : (row + 1) * blockSizeRows, i*colSize : (i + 1) * colSize]
        b = B[i*colSize : (i + 1) * colSize, col * blockSizeCols : (col + 1) * blockSizeCols]
        c = a @ b
        C[row * blockSizeRows:(row + 1) * blockSizeRows, col * blockSizeCols:(col + 1) * blockSizeCols] += c
  """

  acc = tl.zeros((blockSizeRows, blockSizeCols), dtype=tl.float32)
  colSizeCount = tl.cdiv(In1Cols, colSize)


  In1PtrRows = row + tl.arange(0, blockSizeRows)[:, None]
  In2PtrCols = col + tl.arange(0, blockSizeCols)[None, :]
  for i in range(colSizeCount + 1):
    i *= colSize
    In1PtrCols = (i + tl.arange(0, colSize))[None, :]
    In2PtrRows = (i + tl.arange(0, colSize))[:, None] 

    In1Mask = (In1PtrRows < In1Rows) & (In1PtrCols < In1Cols)
    In2Mask = (In2PtrRows < In2Rows) & (In2PtrCols < In2Cols)

    In1Data = tl.load(In1 + In1PtrRows + In1PtrCols, mask = In1Mask, other = 0)
    In2Data = tl.load(In2 + In2PtrRows + In2PtrCols, mask = In2Mask, other = 0)

    acc += tl.dot(In1Data, In2Data, input_precision = 'ieee')
  pass

  OutPtrRows = row + tl.arange(0, blockSizeRows)[:, None]
  OutPtrCols = col + tl.arange(0, blockSizeCols)[None, :]
  OutMask = (OutPtrRows < OutRows) & (OutPtrCols < OutCols)

  tl.store(Out + OutPtrRows + OutPtrCols, acc, mask=OutMask)
pass


def matmul(A: torch.Tensor, B: torch.Tensor, blockSize = (32, 32), colSize = 32) -> torch.Tensor:
  assert A.dtype == B.dtype, "The dtypes of A and B differ"
  assert A.device == B.device, "A and B both should be on the same device"
  assert A.shape[1] == B.shape[0], "Column lengths do not match"
  C = torch.zeros((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

  gridSize = lambda meta: (ceil(C.shape[0] / meta['blockSizeRows']), ceil(C.shape[1] / meta['blockSizeCols']))
  # gridSize = (ceil(C.shape[0] / blockSize[0]), ceil(C.shape[1] / blockSize[1]))
  _matmul_[gridSize](
    A, B, C,
    A.shape[0], A.shape[1],
    B.shape[0], B.shape[1],
    B.shape[0], C.shape[1],
    blockSize[0], blockSize[1],
    colSize,
  )

  return C

# @triton.autotune(configs=[
#   triton.Config({ "blockSizeRows": 256, "blockSizeCols": 256, "colSize": 256, }),
#   triton.Config({ "blockSizeRows": 128, "blockSizeCols": 256, "colSize": 256, }),
#   triton.Config({ "blockSizeRows": 256, "blockSizeCols": 128, "colSize": 256, }),
# ], key = [
#   'OutRows', 'OutCols',
# ])
@triton.jit
def _matmul_pid_optimized_(
  In1, In2, Out,
  In1Rows, In1Cols,
  In2Rows, In2Cols,
  OutRows, OutCols,

  groupRowsSize: tl.constexpr,
  groupColsSize: tl.constexpr,

  blockSizeRows: tl.constexpr,
  blockSizeCols: tl.constexpr,
  colSize      : tl.constexpr,
):
  #pdb.set_trace()
  groupId = tl.program_id(0)
  groupRow = groupId // groupRowsSize
  groupCol = groupId %  groupRowsSize

  tileId = tl.program_id(1)
  tileRow = tileId // OutCols
  tileCol = tileId %  OutCols

  row = (tileRow * groupRowsSize + groupRow) * blockSizeRows
  col = (tileCol * groupColsSize + groupCol) * blockSizeCols

  acc = tl.zeros((blockSizeRows, blockSizeCols), dtype=tl.float32)
  colSizeCount = tl.cdiv(In1Cols, colSize)


  In1PtrRows = row + tl.arange(0, blockSizeRows)[:, None]
  In2PtrCols = col + tl.arange(0, blockSizeCols)[None, :]
  for i in range(colSizeCount + 1):
    i *= colSize
    In1PtrCols = (i + tl.arange(0, colSize))[None, :]
    In2PtrRows = (i + tl.arange(0, colSize))[:, None] 

    In1Mask = (In1PtrRows < In1Rows) & (In1PtrCols < In1Cols)
    In2Mask = (In2PtrRows < In2Rows) & (In2PtrCols < In2Cols)

    In1Data = tl.load(In1 + In1PtrRows + In1PtrCols, mask = In1Mask, other = 0)
    In2Data = tl.load(In2 + In2PtrRows + In2PtrCols, mask = In2Mask, other = 0)

    acc += tl.dot(In1Data, In2Data, input_precision = 'ieee')
  pass

  OutPtrRows = row + tl.arange(0, blockSizeRows)[:, None]
  OutPtrCols = col + tl.arange(0, blockSizeCols)[None, :]
  OutMask = (OutPtrRows < OutRows) & (OutPtrCols < OutCols)

  tl.store(Out + OutPtrRows + OutPtrCols, acc, mask=OutMask)
pass

def matmul_pid_optimized(A: torch.Tensor, B: torch.Tensor, blockSize = (32, 32), colSize = 32) -> torch.Tensor:
  assert A.dtype == B.dtype, "The dtypes of A and B differ"
  assert A.device == B.device, "A and B both should be on the same device"
  assert A.shape[1] == B.shape[0], "Column lengths do not match"
  C = torch.zeros((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

  grid = lambda meta: (
        meta['groupRowsSize'] * meta['groupColsSize'], 
        ceil(C.shape[0] / (meta['blockSizeRows'] * meta['groupRowsSize'])) * ceil(C.shape[1] / (meta['blockSizeCols'] * meta['groupColsSize']))
        )

  _matmul_pid_optimized_[grid](
    A, B, C,
    A.shape[0], A.shape[1],
    B.shape[0], B.shape[1],
    B.shape[0], C.shape[1],
    2, 2,
    blockSize[0], blockSize[1], 
    colSize,
  )

  return C


if __name__ == "__main__":
  size = (4096, 4096)
  A = torch.rand(size, device="cuda", dtype=torch.float32)
  B = torch.rand(size, device="cuda", dtype=torch.float32)

  Cref = A @ B
  Cval = matmul_pid_optimized(A, B) 
  torch.isclose(Cref, Cval)
