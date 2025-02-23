from math import ceil

import triton
import torch
import triton.language as tl

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

  acc = tl.zeros((blockSizeCols, blockSizeRows), dtype=tl.float32)
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

  gridSize = (ceil(C.shape[0] / blockSize[0]), ceil(C.shape[1] / blockSize[1]))
  _matmul_[gridSize](
    A, B, C,
    A.shape[0], A.shape[1],
    B.shape[0], B.shape[1],
    B.shape[0], C.shape[1],
    blockSize[0], blockSize[1],
    colSize,
  )

  return C
