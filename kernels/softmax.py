import torch
import triton
import triton.language as tl
from functools import cache
from math import log, exp

DEVICE = torch.device("cuda:0")

@triton.jit
def _fused_softmax_(
    In, Out,
    InRows, InCols,
    InRowStride, InColStride,
    BlockSizeCols: tl.constexpr,
):
    row = tl.program_id(0)
    rowStride = tl.num_programs(0)
    inf = float('inf')
    for i in tl.range(row, InRows, rowStride, num_stages = 2):
        cols = tl.arange(0, BlockSizeCols)
        mask = cols < InCols
        ptrs = row * InRowStride + (cols * InColStride)
        data = tl.load(In + ptrs, mask = mask, other = -inf)
        max = tl.max(data)
        num = tl.exp(data - max)
        sum = tl.sum(num)
        result = num / sum

        tl.store(Out + ptrs, mask = mask, value = result)
    pass
pass

@triton.jit
def _softmax_(
    In, Out,
    InRows, InCols,
    InRowStride, InColStride,
    BlockSizeCols: tl.constexpr,
):
    row = tl.program_id(0)
    rowStride = tl.num_programs(0)
    count = tl.cdiv(InCols, BlockSizeCols)
    inf = float('inf')
    for i in tl.range(row, InRows, rowStride, num_stages = 2):
        max = -inf
        sum = 0.0
        for idx in tl.range(0, count):
            cols = (idx * BlockSizeCols) + tl.arange(0, BlockSizeCols)
            mask = cols < InCols
            ptrs = row * InRowStride + (cols * InColStride)
            data = tl.load(In + ptrs, mask = mask, other = -inf)
            newMax = tl.max(data)
            if (newMax > max):
                sum *= tl.exp(max - newMax)
                max = newMax
            pass
            num = tl.exp(data - max)
            sum += tl.sum(num)
        pass

        for idx in tl.range(0, count):
            cols = (idx * BlockSizeCols) + tl.arange(0, BlockSizeCols)
            mask = cols < InCols
            ptrs = row * InRowStride + (cols * InColStride)
            data = tl.load(In + ptrs, mask = mask, other = -inf)
            num = tl.exp(data - max)
            result = num / sum
            tl.store(Out + ptrs, mask = mask, value = result)
        pass
    pass
pass

@cache
def getSharedMemSize(device_index: int) -> int:
    properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return properties['max_shared_mem']

def softmax(In: torch.Tensor) -> torch.Tensor:
    InView = In.view(-1, In.shape[-1])
    row, cols = InView.shape
    rStride, cStride = InView.stride()
    Out = torch.empty_like(In)
    sharedMemSize = getSharedMemSize(DEVICE.index) // In.element_size();
    blockSize = triton.next_power_of_2(cols)
    if (sharedMemSize > cols):
        grid = lambda meta: (row, )
        _fused_softmax_[grid](
            In, Out,
            row, cols,
            rStride, cStride,
            blockSize,
        )
    else:
        blockSizeCols = int(2 ** int(log(cols, 2)))
        grid = lambda meta: (InView.shape[-1], )
        _softmax_[grid](
            In, Out,
            row, cols,
            rStride, cStride,
            blockSizeCols, 
        )

    return Out.reshape(In.shape)

if __name__ == "__main__":
    def tester(dataIn): 
        ref = torch.nn.functional.softmax(dataIn, dim=1)
        res = softmax(dataIn)
        if not (torch.allclose(ref, res)):
            print("Value Diff")
            print(ref[:, :5])
            print(res[:, :5])
        else:
            print("Same")

    dataIn = torch.randn(12, 12, device=DEVICE)
    tester(dataIn)
    dataIn = torch.randn(2, 50_000, device=DEVICE)
    tester(dataIn)
    dataIn = torch.ones(2, 50_000, device=DEVICE)
    tester(dataIn)

