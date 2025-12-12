#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_print.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;


// This function body is for SpGEMM only.
// There are assertions that will make it fail for dense matrices,
// but still be cautious and check that you're using sgemm_tcu, not this.
void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pAf = reinterpret_cast<uint8_t *>(arg->Af_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = DIM_M;
  uint32_t N = DIM_N;
  uint32_t K = DIM_K;

  ctx::sparse_fragment_a   fragA;
  ctx::fragment_b          fragB;
  ctx::fragment_acc        fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y *  ctx::tileM;
  uint32_t tile_col = blockIdx.x *  ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

  // tileK = 32
  // Next tile is 32 * 2 = 64 B, but it's actually (32 / 2) * 2 = 32, 0x20
  // 0x40 per row
  // 0x40 * 8 = 0x200

  //if (blockIdx.y == 0 && blockIdx.x == 0) {
  for (int i = 0; i < K; i += ctx::tileK) {
    // Load A tile
    auto pTileA = pA + tile_row * (K / 2) + (i / 2); // The matrix has K / 2 elements for each row.
    //vx_printf("blockIdx.(x, y) = (%d, %d), iter i %d, pTileAA = %p \n", blockIdx.x, blockIdx.y, i, pTileA);
    auto sparsityA = pAf + tile_row * K / 4 + i / 4;
    ctx::load_sparse_matrix_sync(fragA, pTileA, K / 2);

    // Load B tile
    auto pTileB = pB + i * N + tile_col;
    //vx_printf("blockIdx.(x, y) = (%d, %d), iter i %d, pTileB = %p \n", blockIdx.x, blockIdx.y, i, pTileB);
    ctx::load_dense_matrix_sync(fragB, pTileB, N);

    
    // Matrix multiply-accumulate: c += a * b 
    //vx_printf("%d \n", sparsity_pattern);
    ctx::mma_sparse_sync(fragC, fragA, fragB, fragC, *sparsityA);
  }

  // Store the computed C tile
  
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
  
  //} // This thing for block (0, 0) debugging

}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
