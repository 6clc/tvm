extern "C" __global__ void __launch_bounds__(128) conv2d_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_reindex_pad_local[16];
  __shared__ float pad_temp_reindex_pad_shared[576];
  __shared__ float B_reindex_pad_shared[1152];
  for (int64_t var = 0; var < (int64_t)1; ++var) {
    for (int ax2_3_init = 0; ax2_3_init < 4; ++ax2_3_init) {
      for (int ax1_3_init = 0; ax1_3_init < 4; ++ax1_3_init) {
        conv2d_nchw_reindex_pad_local[((ax1_3_init * 4) + ax2_3_init)] = 0.000000e+00f;
      }
    }
    __syncthreads();
    for (int ax0_ax1_ax2_fused_2 = 0; ax0_ax1_ax2_fused_2 < 2; ++ax0_ax1_ax2_fused_2) {
      for (int ax0_ax1_ax2_fused_3_s = 0; ax0_ax1_ax2_fused_3_s < 2; ++ax0_ax1_ax2_fused_3_s) {
        pad_temp_reindex_pad_shared[(((((int)threadIdx.y) * 36) + ((((int)threadIdx.x) >> 2) * 18)) + ((((((int)threadIdx.x) * 4) + (ax0_ax1_ax2_fused_2 * 2)) + ax0_ax1_ax2_fused_3_s) & 15))] = ((((((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) < 49) && ((((int)threadIdx.x) & 3) < 1)) ? A[((((((((int)threadIdx.x) & 3) * 16) + (((((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) / 7) * 8)) + (ax0_ax1_ax2_fused_2 * 8)) + ax0_ax1_ax2_fused_3_s) + ((((((int)blockIdx.x) * 32) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 2)) % 7))] : 0.000000e+00f);
      }
    }
    for (int ax0_ax1_ax2_fused_2_1 = 0; ax0_ax1_ax2_fused_2_1 < 4; ++ax0_ax1_ax2_fused_2_1) {
      for (int ax0_ax1_ax2_fused_3_s_1 = 0; ax0_ax1_ax2_fused_3_s_1 < 2; ++ax0_ax1_ax2_fused_3_s_1) {
        B_reindex_pad_shared[(((((int)threadIdx.y) * 72) + ((((int)threadIdx.x) >> 1) * 18)) + ((((((int)threadIdx.x) * 8) + (ax0_ax1_ax2_fused_2_1 * 2)) + ax0_ax1_ax2_fused_3_s_1) & 15))] = (((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 2)) < 1) && ((((((int)threadIdx.x) & 1) * 2) + (ax0_ax1_ax2_fused_2_1 >> 1)) < 1)) ? B[(((((((int)threadIdx.y) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) >> 1) * 4)) + (ax0_ax1_ax2_fused_2_1 * 2)) + ax0_ax1_ax2_fused_3_s_1)] : 0.000000e+00f);
      }
    }
    __syncthreads();
    for (int ax3_1 = 0; ax3_1 < 16; ++ax3_1) {
      for (int ax2_3 = 0; ax2_3 < 4; ++ax2_3) {
        for (int ax1_3 = 0; ax1_3 < 4; ++ax1_3) {
          conv2d_nchw_reindex_pad_local[((ax1_3 * 4) + ax2_3)] = (conv2d_nchw_reindex_pad_local[((ax1_3 * 4) + ax2_3)] + (pad_temp_reindex_pad_shared[(((((int)threadIdx.x) * 72) + (ax1_3 * 18)) + ax3_1)] * B_reindex_pad_shared[(((((int)threadIdx.y) * 72) + (ax2_3 * 18)) + ax3_1)]));
        }
      }
    }
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      for (int ax2_0 = 0; ax2_0 < 2; ++ax2_0) {
        for (int ax2_1_s = 0; ax2_1_s < 2; ++ax2_1_s) {
          if ((((((int)threadIdx.y) * 2) + ax2_0) < 1) && ((((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 4)) + ax1) < 49)) {
            conv2d_nchw[((((((((int)threadIdx.y) * 196) + (ax2_0 * 98)) + (ax2_1_s * 49)) + (((int)blockIdx.x) * 32)) + (((int)threadIdx.x) * 4)) + ax1)] = conv2d_nchw_reindex_pad_local[(((ax1 * 4) + (ax2_0 * 2)) + ax2_1_s)];
          }
        }
      }
    }
  }
}




