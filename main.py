import torch as T
import triton 
from triton.testing import do_bench
import kernels as K

DEVICE = "cuda"
TYPE = T.float32

def benchmarkSoftmax():
  def run(A: T.Tensor):
    print(f"Running for {A.shape}")
    refSoftmax = lambda x: T.nn.functional.softmax(x, dim = 1)
    T.isclose(refSoftmax (A), K.softmax(A))
    result_normal = do_bench(lambda : refSoftmax(A))
    print (f"normal = {result_normal}")
    result_fused = do_bench(lambda : K.softmax(A))
    print (f"fused = {result_fused}")
    
  for i in range(1, 100, 10):
    A = T.rand((1, 256 * i), device=DEVICE, dtype=TYPE)
    run(A)

def benchmarkMatmul():
  A = T.rand((4096, 4096), device=DEVICE, dtype=TYPE)
  B = T.rand((4096, 4096), device=DEVICE, dtype=TYPE)
  T.isclose(A@B, K.matmul(A, B))

  result_normal = do_bench(lambda : A @ B)
  print (f"normal = {result_normal}")

  # XXX: Buggy
  # print("Ref Optimized")
  # for i in range(4, 8):
  #   try:
  #     result_custom = do_bench(lambda: K.matmul_ref(A, B))
  #     print(f"Ref {i} = {result_custom}")
  #   except Exception as e:
  #     print(f"Ref  {i} = {str(e)}")

  print("Naive")
  for i in range(4, 8):
    try:
      result_custom = do_bench(lambda: K.matmul(A, B, blockSize=(2 ** i, 2 ** i)))
      print(f"Custom {i} = {result_custom}")
    except Exception as e:
      print(f"Custom {i} = {repr(e)}")
  
  print("PID Optimized")
  for i in range(4, 8):
    try:
      result_custom = do_bench(lambda: K.matmul_pid_optimized(A, B, blockSize=(2 ** i, 2 ** i)))
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

def benchmarkVectorAdd2D():
  A = T.rand((4096, 4096), device=DEVICE, dtype=TYPE)
  B = T.rand((4096, 4096), device=DEVICE, dtype=TYPE)
  T.isclose(A + B, K.vectorAdd2D(A, B))

  result_normal = do_bench(lambda : A + B)
  print (f"normal = {result_normal}")

  for i in range(4, 9):
    try:
      result_custom = do_bench(lambda: K.vectorAdd2D(A, B))
      print(f"Custom {i} = {result_custom}")
    except Exception as e:
      print(f"Custom {i} = {repr(e)}, {str(e)}")

if __name__ == '__main__':
  benchmarkFns = [
    # benchmarkMatmul,
    benchmarkSoftmax,
    benchmarkVectorAdd,
    benchmarkVectorAdd2D,
  ]
  for fn in benchmarkFns:
    print(f"Running: {fn.__name__}")
    fn()
