#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <vector>

#ifndef DIM_N
#define DIM_N 64
#endif

#ifndef DIM_M
#define DIM_M DIM_N
#endif

#ifndef DIM_K
#define DIM_K DIM_N
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t Af_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
