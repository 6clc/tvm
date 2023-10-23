from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R



@T.prim_func(private=True)
def main(A: T.Buffer((T.int64(1), T.int64(1), T.int64(256), T.int64(256)), "float32"), B: T.Buffer((T.int64(64), T.int64(1), T.int64(4), T.int64(4)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(253), T.int64(253)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(256), T.int64(256)))
    pad_temp_reindex = T.alloc_buffer((T.int64(1), T.int64(64009), T.int64(16)))
    B_reindex = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(16)))
    conv2d_nchw_reindex = T.alloc_buffer((T.int64(1), T.int64(64009), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(256), T.int64(256)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(T.int64(1), T.int64(253), T.int64(253), T.int64(1), T.int64(4), T.int64(4)):
        with T.block("pad_temp_reindex_reindex"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(pad_temp[v0, v3, v1 + v4, v2 + v5])
            T.writes(pad_temp_reindex[T.int64(0), v1 * T.int64(253) + v2, v4 * T.int64(4) + v5])
            pad_temp_reindex[T.int64(0), v1 * T.int64(253) + v2, v4 * T.int64(4) + v5] = pad_temp[v0, v3, v1 + v4, v2 + v5]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(1), T.int64(4), T.int64(4)):
        with T.block("B_reindex_reindex"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v0, v1, v2, v3])
            T.writes(B_reindex[T.int64(0), v0, v2 * T.int64(4) + v3])
            B_reindex[T.int64(0), v0, v2 * T.int64(4) + v3] = B[v0, v1, v2, v3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(64), T.int64(253), T.int64(253), T.int64(1), T.int64(4), T.int64(4)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp_reindex[T.int64(0), v_yy * T.int64(253) + v_xx, v_ry * T.int64(4) + v_rx], B_reindex[T.int64(0), v_ff, v_ry * T.int64(4) + v_rx])
            T.writes(conv2d_nchw_reindex[T.int64(0), v_yy * T.int64(253) + v_xx, v_ff])
            with T.init():
                conv2d_nchw_reindex[T.int64(0), v_yy * T.int64(253) + v_xx, v_ff] = T.float32(0)
            conv2d_nchw_reindex[T.int64(0), v_yy * T.int64(253) + v_xx, v_ff] = conv2d_nchw_reindex[T.int64(0), v_yy * T.int64(253) + v_xx, v_ff] + pad_temp_reindex[T.int64(0), v_yy * T.int64(253) + v_xx, v_ry * T.int64(4) + v_rx] * B_reindex[T.int64(0), v_ff, v_ry * T.int64(4) + v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(253), T.int64(253)):
        with T.block("conv2d_nchw_reindex"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(conv2d_nchw_reindex[T.int64(0), v2 * T.int64(253) + v3, v1])
            T.writes(conv2d_nchw[v0, v1, v2, v3])
            conv2d_nchw[v0, v1, v2, v3] = conv2d_nchw_reindex[T.int64(0), v2 * T.int64(253) + v3, v1]


@T.prim_func(private=True)
def main(A: T.Buffer((T.int64(1), T.int64(1), T.int64(256), T.int64(256)), "float32"), B: T.Buffer((T.int64(64), T.int64(1), T.int64(4), T.int64(4)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(253), T.int64(253)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(256), T.int64(256)))
    pad_temp_reindex = T.alloc_buffer((T.int64(1), T.int64(64009), T.int64(16)))
    B_reindex = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(16)))
    conv2d_nchw_reindex = T.alloc_buffer((T.int64(1), T.int64(64009), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(256), T.int64(256)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(T.int64(1), T.int64(253), T.int64(253), T.int64(1), T.int64(4), T.int64(4)):
        with T.block("pad_temp_reindex_reindex"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(pad_temp[v0, v3, v1 + v4, v2 + v5])
            T.writes(pad_temp_reindex[T.int64(0), v1 * T.int64(253) + v2, v4 * T.int64(4) + v5])
            pad_temp_reindex[T.int64(0), v1 * T.int64(253) + v2, v4 * T.int64(4) + v5] = pad_temp[v0, v3, v1 + v4, v2 + v5]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(1), T.int64(4), T.int64(4)):
        with T.block("B_reindex_reindex"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v0, v1, v2, v3])
            T.writes(B_reindex[T.int64(0), v0, v2 * T.int64(4) + v3])
            B_reindex[T.int64(0), v0, v2 * T.int64(4) + v3] = B[v0, v1, v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64009), T.int64(64), T.int64(16)):
        with T.block("conv2d_nchw"):
            v0, v1, v2, v3 = T.axis.remap("SSSR", [ax0, ax1, ax2, ax3])
            T.reads(pad_temp_reindex[T.int64(0), v1, v3], B_reindex[T.int64(0), v2, v3])
            T.writes(conv2d_nchw_reindex[T.int64(0), v1, v2])
            with T.init():
                conv2d_nchw_reindex[T.int64(0), v1, v2] = T.float32(0)
            conv2d_nchw_reindex[T.int64(0), v1, v2] = conv2d_nchw_reindex[T.int64(0), v1, v2] + pad_temp_reindex[T.int64(0), v1, v3] * B_reindex[T.int64(0), v2, v3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(64), T.int64(253), T.int64(253)):
        with T.block("conv2d_nchw_reindex"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(conv2d_nchw_reindex[T.int64(0), v2 * T.int64(253) + v3, v1])
            T.writes(conv2d_nchw[v0, v1, v2, v3])
            conv2d_nchw[v0, v1, v2, v3] = conv2d_nchw_reindex[T.int64(0), v2 * T.int64(253) + v3, v1]

