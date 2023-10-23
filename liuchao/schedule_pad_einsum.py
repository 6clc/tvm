from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import tvm
@T.prim_func
def before_pad_einsum(
    A: T.Buffer((127, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((127, 127), "float32"),
) -> None:
    for i0, i1, i2 in T.grid(127, 127, 127):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]

sch = tvm.tir.Schedule(before_pad_einsum, debug_mask="all")
block = sch.get_block("C_shared")
sch.pad_einsum(block, [128, 1, 1])
print(sch.mod["main"].script())
