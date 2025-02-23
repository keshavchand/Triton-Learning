import torch as T
import triton 
from triton.testing import do_bench
import kernels as K

DEVICE = "cuda"
TYPE = T.float32

def benchmarkMatmul():
  A = T.rand((4096, 4096), device=DEVICE, dtype=TYPE)
  B = T.rand((4096, 4096), device=DEVICE, dtype=TYPE)
  T.isclose(A@B, K.matmul(A, B))

  result_normal = do_bench(lambda : A @ B)
  print (f"normal = {result_normal}")

  for i in range(4, 8):
    try:
      result_custom = do_bench(lambda: K.matmul(A, B, blockSize=(2 ** i, 2 ** i)))
      print(f"Custom {i} = {result_custom}")
    except Exception as e:
      print(f"Custom {i} = {repr(e)}")

def benchmarkVectorAdd():
  A = T.rand((4096 * 4096), device=DEVICE, dtype=TYPE)
  B = T.rand((4096 * 4096), device=DEVICE, dtype=TYPE)
  T.isclose(A + B, K.vectorAdd(A, B))

  result_normal = do_bench(lambda : A + B)
  print (f"normal = {result_normal}")

  for i in range(4, 9):
    try:
      result_custom = do_bench(lambda: K.vectorAdd(A, B, blockSize=(2 ** i,)))
      print(f"Custom {i} = {result_custom}")
    except Exception as e:
      print(f"Custom {i} = {repr(e)}, {str(e)}")


if __name__ == '__main__':
  benchmarkFns = [
    benchmarkMatmul,
    benchmarkVectorAdd,
  ]
  for fn in benchmarkFns:
    print(f"Running: {fn.__name__}")
    fn()
