"""
Week 5: FlashAttention Algorithm 1 reimplemented with CuTe.
Run:
    python -m modal run flash_attention_cute_modal.py
"""

import modal

app = modal.App("flash-attention-cute-week5")

cuda_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .run_commands(
        "git clone --depth 1 --branch v3.5.0 "
        "https://github.com/NVIDIA/cutlass.git /cutlass"
    )
)

CUDA_SRC = r"""
/*
 * FlashAttention Algorithm 1 — CuTe reimplementation
 *
 * CuTe concepts:
 *   Layout     = (Shape, Stride)  idx = inner_product(coord, stride)
 *   Tensor     = make_tensor(ptr, layout)  — zero-copy view over memory
 *   local_tile = extract the j-th tile of tile_shape from a Tensor
 *
 * Grid  : Tr blocks  (one per kBr-row-tile of Q)
 * Block : kBc threads
 * All matrices row-major: (N, kD):(kD, 1)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "cute/tensor.hpp"
#include "cute/layout.hpp"
using namespace cute;

static constexpr int kBr = 32;
static constexpr int kBc = 32;
static constexpr int kD  = 64;

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  exit(1);} } while(0)

__global__ void flash_attention_cute_kernel(
    const float* __restrict__ Q_ptr,
    const float* __restrict__ K_ptr,
    const float* __restrict__ V_ptr,
          float* __restrict__ O_ptr,
          float* __restrict__ L_ptr,
    int N)
{
    /* ── 1. Global Tensors (HBM views) ───────────────────────────────────
     * Layout (N, kD):(kD, 1)  →  idx(r,c) = r*kD + c
     * make_tensor binds the pointer to that layout — no data is copied.  */
    auto g_layout = make_layout(make_shape (N,        Int<kD>{}),
                                make_stride(Int<kD>{}, Int<1>{}));

    Tensor gQ = make_tensor(make_gmem_ptr(Q_ptr), g_layout);
    Tensor gK = make_tensor(make_gmem_ptr(K_ptr), g_layout);
    Tensor gV = make_tensor(make_gmem_ptr(V_ptr), g_layout);
    Tensor gO = make_tensor(make_gmem_ptr(O_ptr), g_layout);

    /* ── 2. This block's Q / O tile ─────────────────────────────────────
     * local_tile(tensor, tile_shape, tile_coord) returns a zero-copy
     * sub-Tensor.  tQi(r,c) == gQ(blockIdx.x*kBr + r, c).              */
    auto tile_QO = make_shape(Int<kBr>{}, Int<kD>{});
    auto coord_i = make_coord(blockIdx.x, 0);
    Tensor tQi = local_tile(gQ, tile_QO, coord_i);
    Tensor tOi = local_tile(gO, tile_QO, coord_i);

    int row_start = blockIdx.x * kBr;
    int actual_Br = min(kBr, N - row_start);
    if (actual_Br <= 0) return;

    /* ── 3. Shared memory ────────────────────────────────────────────────
     * SMEM floats = 2*kBr*kD + 2*kBc*kD + kBr*kBc + 2*kBr
     *             = 2*32*64 + 2*32*64 + 32*32 + 2*32
     *             = 4096 + 4096 + 1024 + 64 = 9280 floats = 36.25 KB   */
    extern __shared__ float smem[];
    float* raw_Qi  = smem;
    float* raw_Kj  = raw_Qi  + kBr * kD;
    float* raw_Vj  = raw_Kj  + kBc * kD;
    float* raw_Sij = raw_Vj  + kBc * kD;
    float* raw_Oi  = raw_Sij + kBr * kBc;
    float* raw_mi  = raw_Oi  + kBr * kD;
    float* raw_li  = raw_mi  + kBr;

    /* CuTe SMEM Tensors — same Layout abstraction, different pointer type */
    Tensor sQi  = make_tensor(make_smem_ptr(raw_Qi),
                    make_layout(make_shape (Int<kBr>{}, Int<kD>{}),
                                make_stride(Int<kD>{},  Int<1>{})));
    Tensor sKj  = make_tensor(make_smem_ptr(raw_Kj),
                    make_layout(make_shape (Int<kBc>{}, Int<kD>{}),
                                make_stride(Int<kD>{},  Int<1>{})));
    Tensor sVj  = make_tensor(make_smem_ptr(raw_Vj),
                    make_layout(make_shape (Int<kBc>{}, Int<kD>{}),
                                make_stride(Int<kD>{},  Int<1>{})));
    Tensor sSij = make_tensor(make_smem_ptr(raw_Sij),
                    make_layout(make_shape (Int<kBr>{},  Int<kBc>{}),
                                make_stride(Int<kBc>{}, Int<1>{})));
    Tensor sOi  = make_tensor(make_smem_ptr(raw_Oi),
                    make_layout(make_shape (Int<kBr>{}, Int<kD>{}),
                                make_stride(Int<kD>{},  Int<1>{})));

    int tid = threadIdx.x; // [0, kBc)

    /* ── 4. Load Qi into SMEM & init accumulators ────────────────────── */
    for (int r = 0; r < actual_Br; r++)
        for (int c = tid; c < kD; c += kBc)
            sQi(r, c) = tQi(r, c);

    for (int r = tid; r < actual_Br; r += kBc) {
        raw_mi[r] = -FLT_MAX;
        raw_li[r] = 0.0f;
        for (int c = 0; c < kD; c++) sOi(r, c) = 0.0f;
    }
    __syncthreads();

    /* ── 5. Outer loop over KV tiles ─────────────────────────────────── */
    float scale   = rsqrtf((float)kD);
    int   Tc      = (N + kBc - 1) / kBc;
    auto  tile_KV = make_shape(Int<kBc>{}, Int<kD>{});

    for (int j = 0; j < Tc; j++) {
        auto coord_j = make_coord(j, 0);

        /* local_tile: zero-copy (kBc, kD) view into gK / gV */
        Tensor tKj = local_tile(gK, tile_KV, coord_j);
        Tensor tVj = local_tile(gV, tile_KV, coord_j);

        int actual_Bc = min(kBc, N - j * kBc);
        if (actual_Bc <= 0) break;

        /* 5a. Load Kj, Vj  HBM → SMEM */
        for (int r = 0; r < actual_Bc; r++)
            for (int c = tid; c < kD; c += kBc) {
                sKj(r, c) = tKj(r, c);
                sVj(r, c) = tVj(r, c);
            }
        __syncthreads();

        /* 5b. Sij = scale * Qi * Kj^T
         * Thread tid fills column tid of sSij for all rows.             */
        if (tid < actual_Bc) {
            for (int r = 0; r < actual_Br; r++) {
                float acc = 0.0f;
                for (int c = 0; c < kD; c++)
                    acc += sQi(r, c) * sKj(tid, c);
                sSij(r, tid) = acc * scale;
            }
        }
        __syncthreads();

        /* 5c-d. Online softmax + accumulate (Algorithm 1 lines 10-12)
         * Each thread owns rows r = tid, tid+kBc, ...
         * Serial over rows, fully local — no extra syncs needed.        */
        for (int r = tid; r < actual_Br; r += kBc) {
            float mi_old = raw_mi[r];
            float li_old = raw_li[r];

            /* new row-max */
            float mi_new = mi_old;
            for (int k = 0; k < actual_Bc; k++)
                mi_new = fmaxf(mi_new, sSij(r, k));

            /* new normaliser */
            float li_new = expf(mi_old - mi_new) * li_old;
            for (int k = 0; k < actual_Bc; k++)
                li_new += expf(sSij(r, k) - mi_new);

            /* rescale running Oi then accumulate P*V — no extra sync  */
            float alpha = expf(mi_old - mi_new) * li_old;
            for (int c = 0; c < kD; c++)
                sOi(r, c) *= alpha;

            for (int k = 0; k < actual_Bc; k++) {
                float p = expf(sSij(r, k) - mi_new);
                for (int c = 0; c < kD; c++)
                    sOi(r, c) += p * sVj(k, c);
            }

            for (int c = 0; c < kD; c++)
                sOi(r, c) /= li_new;

            raw_mi[r] = mi_new;
            raw_li[r] = li_new;
        }
        __syncthreads();
    }

    /* ── 6. Write Oi and Li back to HBM ─────────────────────────────── */
    for (int r = 0; r < actual_Br; r++)
        for (int c = tid; c < kD; c += kBc)
            tOi(r, c) = sOi(r, c);

    for (int r = tid; r < actual_Br; r += kBc)
        L_ptr[row_start + r] = raw_mi[r] + logf(raw_li[r] + 1e-20f);
}

int main() {
    const int N = 512;

    printf("======================================================\n");
    printf("  FlashAttention - CuTe (Week 5)\n");
    printf("  N=%d  d=%d  kBr=%d  kBc=%d\n", N, kD, kBr, kBc);
    printf("======================================================\n");

    size_t smem = (2ull*kBr*kD + 2ull*kBc*kD + kBr*kBc + 2ull*kBr)
                  * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device : %s\n", prop.name);
    printf("  SMEM   : %.1f KB needed / %.1f KB available\n\n",
           smem/1024.f, prop.sharedMemPerBlock/1024.f);
    if (smem > prop.sharedMemPerBlock) {
        printf("ERROR: not enough shared memory\n"); return 1;
    }

    size_t mat = (size_t)N * kD * sizeof(float);
    size_t vec = (size_t)N * sizeof(float);

    float *hQ=(float*)malloc(mat), *hK=(float*)malloc(mat);
    float *hV=(float*)malloc(mat), *hO=(float*)malloc(mat);
    float *hL=(float*)malloc(vec);

    srand(42);
    for (int i = 0; i < N*kD; i++) {
        hQ[i] = ((float)rand()/RAND_MAX)*2.f-1.f;
        hK[i] = ((float)rand()/RAND_MAX)*2.f-1.f;
        hV[i] = ((float)rand()/RAND_MAX)*2.f-1.f;
    }

    float *dQ,*dK,*dV,*dO,*dL;
    CUDA_CHECK(cudaMalloc(&dQ,mat)); CUDA_CHECK(cudaMalloc(&dK,mat));
    CUDA_CHECK(cudaMalloc(&dV,mat)); CUDA_CHECK(cudaMalloc(&dO,mat));
    CUDA_CHECK(cudaMalloc(&dL,vec));
    CUDA_CHECK(cudaMemcpy(dQ,hQ,mat,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,hK,mat,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,mat,cudaMemcpyHostToDevice));

    int Tr = (N + kBr - 1) / kBr;
    printf("  Grid : %d blocks x %d threads\n\n", Tr, kBc);

    flash_attention_cute_kernel<<<Tr, kBc, smem>>>(dQ,dK,dV,dO,dL,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hO,dO,mat,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL,dL,vec,cudaMemcpyDeviceToHost));

    printf("  Completed successfully!\n");
    printf("  O[0][0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           hO[0],hO[1],hO[2],hO[3],hO[4]);
    printf("  L[0:5]    = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           hL[0],hL[1],hL[2],hL[3],hL[4]);
    printf("======================================================\n");

    free(hQ);free(hK);free(hV);free(hO);free(hL);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dL);
    return 0;
}
"""

@app.function(gpu="T4", image=cuda_image, timeout=600)
def run_flash_attention_cute():
    import subprocess, tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "fa_cute.cu")
        exe = os.path.join(tmp, "fa_cute")

        with open(src, "w") as f:
            f.write(CUDA_SRC)

        cmd = ["nvcc", "-O2", "-std=c++17", "-arch=sm_75",
               "-I/cutlass/include", src, "-o", exe, "-lm"]
        print("Compiling...\n", " ".join(cmd), "\n")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("COMPILE ERROR:\n", r.stderr)
            return
        print("Compiled OK\n")

        r = subprocess.run([exe], capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print("RUNTIME ERROR:\n", r.stderr)

@app.local_entrypoint()
def main():
    run_flash_attention_cute.remote()