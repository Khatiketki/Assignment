/*
 * FlashAttention Algorithm 1 — CuTe reimplementation
 * ====================================================
 * Compile:
 *   nvcc -O2 -std=c++17 -arch=sm_75 -I/cutlass/include fa_cute.cu -o fa_cute -lm
 *
 * CuTe concepts used
 * ------------------
 *  Layout     = (Shape, Stride)
 *               idx(coord) = inner_product(coord, stride)
 *               e.g. row-major (N,d):(d,1)  →  idx(r,c) = r*d + c*1
 *
 *  Tensor     = make_tensor(ptr, layout)
 *               Binds a raw pointer to a layout — zero-copy view.
 *               tensor(i, j)  ==  ptr[ i*stride0 + j*stride1 ]
 *
 *  local_tile = local_tile(tensor, tile_shape, tile_coord)
 *               Returns the sub-Tensor starting at tile_coord * tile_shape.
 *               No data moved — CuTe composes the offset into the strides.
 *
 * Grid  : Tr blocks  (one per kBr-row-tile of Q)
 * Block : kBc threads (one per column of the current Kj/Vj tile)
 *
 * All matrices row-major: Q/K/V/O have layout (N, kD):(kD, 1).
 *
 * Shared memory breakdown (kBr=32, kBc=32, kD=64):
 *   sQi  (32,64) + sKj (32,64) + sVj (32,64)
 *   + sSij (32,32) + sOi (32,64) + smi (32) + sli (32)
 *   = (32*64)*3 + 32*32 + 32*64 + 32 + 32
 *   = 11,328 floats * 4 = 44.2 KB  (fits in 48 KB T4 limit)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ── CuTe headers (from CUTLASS) ── */
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

using namespace cute;

/* ── Compile-time tile / dimension constants ── */
static constexpr int kBr = 32;   // rows per Q-tile  (= rows of O-tile)
static constexpr int kBc = 32;   // cols per K/V-tile (= threads per block)
static constexpr int kD  = 64;   // head dimension

/* ════════════════════════════════════════════════════════════════════════════
 * Kernel
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void flash_attention_cute_kernel(
    const float* __restrict__ Q_ptr,   // (N, kD) row-major in HBM
    const float* __restrict__ K_ptr,   // (N, kD) row-major in HBM
    const float* __restrict__ V_ptr,   // (N, kD) row-major in HBM
          float* __restrict__ O_ptr,   // (N, kD) row-major in HBM
          float* __restrict__ L_ptr,   // (N,)    logsumexp
    int N)
{
    /* ── 1. Build global Tensors over HBM ────────────────────────────────
     *
     * Layout: shape=(N, kD), stride=(kD, 1)  →  row-major
     *   idx(row, col) = row*kD + col*1
     *
     * make_tensor(ptr, layout) is a zero-copy view — no data moved.
     * Indexing via tensor(r, c) uses the layout's inner-product rule.
     */
    auto g_layout = make_layout(make_shape (N,        Int<kD>{}),
                                make_stride(Int<kD>{}, Int<1>{}));

    Tensor gQ = make_tensor(make_gmem_ptr(Q_ptr), g_layout); // (N, kD)
    Tensor gK = make_tensor(make_gmem_ptr(K_ptr), g_layout); // (N, kD)
    Tensor gV = make_tensor(make_gmem_ptr(V_ptr), g_layout); // (N, kD)
    Tensor gO = make_tensor(make_gmem_ptr(O_ptr), g_layout); // (N, kD)

    /* ── 2. Extract this block's Q / O row-tile ──────────────────────────
     *
     * local_tile(tensor, tile_shape, tile_coord)
     *   tile_shape = (kBr, kD)
     *   tile_coord = (blockIdx.x, 0)
     *
     * Returns a (kBr, kD) Tensor whose element (r, c) maps to
     *   gQ(blockIdx.x*kBr + r, c)
     * CuTe computes that offset by composing layouts — no manual arithmetic.
     */
    auto tile_QO  = make_shape(Int<kBr>{}, Int<kD>{});
    auto coord_i  = make_coord(blockIdx.x, 0);

    Tensor tQi = local_tile(gQ, tile_QO, coord_i); // (kBr, kD)
    Tensor tOi = local_tile(gO, tile_QO, coord_i); // (kBr, kD)

    int row_start = blockIdx.x * kBr;
    int actual_Br = min(kBr, N - row_start);
    if (actual_Br <= 0) return;

    /* ── 3. Shared memory — one flat array, carved into named regions ────*/
    extern __shared__ float smem[];
    float* raw_Qi  = smem;
    float* raw_Kj  = raw_Qi  + kBr * kD;
    float* raw_Vj  = raw_Kj  + kBc * kD;
    float* raw_Sij = raw_Vj  + kBc * kD;
    float* raw_Oi  = raw_Sij + kBr * kBc;
    float* raw_mi  = raw_Oi  + kBr * kD;
    float* raw_li  = raw_mi  + kBr;

    /* Build CuTe Tensors over SMEM.
     * Same Layout abstraction — just a different pointer type (smem vs gmem).
     *
     * sQi  : (kBr, kD)  stride (kD, 1)   row-major
     * sKj  : (kBc, kD)  stride (kD, 1)   row-major
     * sVj  : (kBc, kD)  stride (kD, 1)   row-major
     * sSij : (kBr, kBc) stride (kBc, 1)  row-major
     * sOi  : (kBr, kD)  stride (kD, 1)   row-major
     */
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

    /* ── 4. Load Qi into SMEM & initialise accumulators ─────────────────
     * Each thread (tid) covers columns tid, tid+kBc, tid+2*kBc, ...
     * sQi(r, c) = tQi(r, c)  — CuTe handles index arithmetic on both sides.
     */
    int tid = threadIdx.x;   // in [0, kBc)

    for (int r = 0; r < actual_Br; r++)
        for (int c = tid; c < kD; c += kBc)
            sQi(r, c) = tQi(r, c);

    for (int r = 0; r < actual_Br; r++) {
        raw_mi[r] = -FLT_MAX;
        raw_li[r] = 0.0f;
        for (int c = 0; c < kD; c++) sOi(r, c) = 0.0f;
    }
    __syncthreads();

    /* ── 5. Outer loop over KV tiles ─────────────────────────────────────*/
    int Tc        = (N + kBc - 1) / kBc;
    auto tile_KV  = make_shape(Int<kBc>{}, Int<kD>{});

    for (int j = 0; j < Tc; j++) {
        auto coord_j = make_coord(j, 0);

        /* local_tile gives the j-th (kBc, kD) view into gK / gV.
         * tKj(r, c)  ==  gK(j*kBc + r, c)  — zero-copy, no data moved.  */
        Tensor tKj = local_tile(gK, tile_KV, coord_j); // (kBc, kD)
        Tensor tVj = local_tile(gV, tile_KV, coord_j); // (kBc, kD)

        int col_start = j * kBc;
        int actual_Bc = min(kBc, N - col_start);
        if (actual_Bc <= 0) break;

        /* 5a. Load Kj, Vj  HBM → SMEM */
        for (int r = 0; r < actual_Bc; r++)
            for (int c = tid; c < kD; c += kBc) {
                sKj(r, c) = tKj(r, c);
                sVj(r, c) = tVj(r, c);
            }
        __syncthreads();

        /* 5b. Sij = Qi * Kj^T
         * Thread tid computes column tid of sSij.
         * sSij(r, tid) = dot( sQi[row r], sKj[row tid] )               */
        if (tid < actual_Bc) {
            for (int r = 0; r < actual_Br; r++) {
                float dot = 0.0f;
                for (int c = 0; c < kD; c++)
                    dot += sQi(r, c) * sKj(tid, c);
                sSij(r, tid) = dot;
            }
        }
        __syncthreads();

        /* 5c-d. Online softmax update (Algorithm 1, lines 10-12)
         * Thread tid owns rows r = tid, tid+kBc, tid+2*kBc, ...        */
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

            /* rescale running Oi */
            float alpha = expf(mi_old - mi_new) * li_old;
            for (int c = 0; c < kD; c++)
                sOi(r, c) *= alpha;

            /* accumulate unnormalised P*V */
            for (int k = 0; k < actual_Bc; k++) {
                float p = expf(sSij(r, k) - mi_new);
                for (int c = 0; c < kD; c++)
                    sOi(r, c) += p * sVj(k, c);
            }

            /* normalise */
            for (int c = 0; c < kD; c++)
                sOi(r, c) /= li_new;

            raw_mi[r] = mi_new;
            raw_li[r] = li_new;
        }
        __syncthreads();
    }

    /* ── 6. Write Oi and Li back to HBM ──────────────────────────────────
     * tOi(r, c) = sOi(r, c)  — CuTe indexes into HBM via the layout.   */
    for (int r = 0; r < actual_Br; r++)
        for (int c = tid; c < kD; c += kBc)
            tOi(r, c) = sOi(r, c);

    for (int r = tid; r < actual_Br; r += kBc)
        L_ptr[row_start + r] = raw_mi[r] + logf(raw_li[r]);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Host
 * ════════════════════════════════════════════════════════════════════════════ */
#define CHECK(x) do {                                                      \
    cudaError_t _e = (x);                                                  \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s line %d: %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));               \
        exit(1);                                                           \
    }                                                                      \
} while(0)

int main()
{
    const int N = 512;

    printf("======================================================\n");
    printf("  FlashAttention – CuTe (Week 5)\n");
    printf("  N=%d  d=%d  kBr=%d  kBc=%d\n", N, kD, kBr, kBc);
    printf("======================================================\n");

    /* shared memory required */
    size_t smem = (size_t)(3*kBr*kD + 2*kBc*kD + kBr*kBc + 2*kBr)
                  * sizeof(float);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device : %s\n", prop.name);
    printf("  SMEM   : %.1f KB needed  /  %.1f KB available\n\n",
           smem/1024.f, prop.sharedMemPerBlock/1024.f);
    if (smem > prop.sharedMemPerBlock) {
        fprintf(stderr, "Not enough shared memory!\n"); return 1;
    }

    /* host allocations */
    size_t mat = (size_t)N * kD * sizeof(float);
    size_t vec = (size_t)N      * sizeof(float);

    float *hQ = (float*)malloc(mat), *hK = (float*)malloc(mat);
    float *hV = (float*)malloc(mat), *hO = (float*)malloc(mat);
    float *hL = (float*)malloc(vec);

    srand(42);
    for (int i = 0; i < N * kD; i++) {
        hQ[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
        hK[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
        hV[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    }

    /* device allocations */
    float *dQ, *dK, *dV, *dO, *dL;
    CHECK(cudaMalloc(&dQ, mat)); CHECK(cudaMalloc(&dK, mat));
    CHECK(cudaMalloc(&dV, mat)); CHECK(cudaMalloc(&dO, mat));
    CHECK(cudaMalloc(&dL, vec));

    CHECK(cudaMemcpy(dQ, hQ, mat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dK, hK, mat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dV, hV, mat, cudaMemcpyHostToDevice));

    /* launch */
    int Tr = (N + kBr - 1) / kBr;
    printf("  Grid : %d blocks x %d threads\n\n", Tr, kBc);

    flash_attention_cute_kernel<<<Tr, kBc, smem>>>(dQ, dK, dV, dO, dL, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    /* copy back */
    CHECK(cudaMemcpy(hO, dO, mat, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hL, dL, vec, cudaMemcpyDeviceToHost));

    printf("  Completed successfully!\n");
    printf("  O[0][0:5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           hO[0], hO[1], hO[2], hO[3], hO[4]);
    printf("  L[0:5]    = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           hL[0], hL[1], hL[2], hL[3], hL[4]);
    printf("======================================================\n");

    /* cleanup */
    free(hQ); free(hK); free(hV); free(hO); free(hL);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
    return 0;
}