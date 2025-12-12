#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP 1000
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;

template <typename T>
struct data_accessor_t {
  using Type = typename T::dtype;
  static Type read(const Type *ptr, uint32_t offset) {
    return ptr[offset];
  }
  static void write(Type *ptr, uint32_t offset, Type value) {
    ptr[offset] = value;
  }
};


template <typename Type>
class Comparator {};

template <>
class Comparator<vt::int8> {
public:
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::uint8> {
public:
  static uint8_t generate() {
    return (uint8_t)rand();
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int32> {
public:
  static int32_t generate() {
    return (int32_t)rand();
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftoh_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::bf16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftob_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, fb.f, fa.f);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename S, typename D>
struct muladd_t {
  using stype = typename S::dtype;
  using dtype = typename D::dtype;
  static dtype eval(stype a, stype b, dtype c) {
    return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
  }
};

template <>
struct muladd_t<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_htof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftoh_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

template <>
struct muladd_t<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_btof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_btof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_btof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_btof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_btof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftob_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

///////////////////////////////////////////////////////////////////////////////

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE, 4, 8, 0, true>;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  uint32_t subbytes = 8 / vt::ITYPE::bits;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum(0);
      for (uint32_t k = 0; k < KS; ++k) {
        auto a = data_accessor_t<vt::ITYPE>::read(A, m * KS + k);
        auto b = data_accessor_t<vt::ITYPE>::read(B, k * N + n);
        sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a, b, sum);
      }
      data_accessor_t<vt::OTYPE>::write(C, m * N + n, sum);
    }
  }
}

struct SparseMat {
  std::vector<itype_t> values;   // non-zeros
  std::vector<uint8_t> meta;     // Array of sparsity patterns

  uint32_t rows, cols;           // original A dims (M × K)
};

enum SparsityPattern {
    P0011 = 0b0011,
    P0101 = 0b0101,
    P1001 = 0b1001,
    P0110 = 0b0110,
    P1010 = 0b1010,
    P1100 = 0b1100
};


// This was mostly adapted from the original code in sgemm_tcu.
// The justification for keeping the same pruning strategy is mentioned in the slides.
static void prune_dense_matrix(std::vector<itype_t>& denseA, uint32_t N) {
  itype_t* src = denseA.data();
  for (uint32_t i = 0; i < N; i += 4) {
    itype_t blk[4] = {src[i], src[i+1], src[i+2], src[i+3]};

    uint32_t idx[4] = {0, 1, 2, 3};
    std::sort(idx, idx + 4,
        [&](uint32_t a, uint32_t b) {
          return std::abs((int)idx[a]) < std::abs((int)idx[b]);
        });

    uint8_t erase0 = idx[2];
    uint8_t erase1 = idx[3];

    blk[erase0] = 0;
    blk[erase1] = 0;

    src[i]   = blk[0];
    src[i+1] = blk[1];
    src[i+2] = blk[2];
    src[i+3] = blk[3];
  }
}

static SparseMat compress_sparse_matrix(const std::vector<itype_t>& sparseA, uint32_t M, uint32_t K) {
  SparseMat out;
  out.rows = M;
  out.cols = K;
  // 2 out of 4 contiguous values
  out.values.reserve(M * K); 

  // 1 entry for every 4 elements
  out.meta.reserve(M * K / 4);

  const itype_t* src = sparseA.data();
  uint32_t N = M * K;

  for (uint32_t i = 0; i < N; i += 4) {
    uint8_t pattern = 0;

    itype_t blk[4] = {
        src[i + 0],
        src[i + 1],
        src[i + 2],
        src[i + 3]
    };

    bool blk0_zero = blk[0] == 0;
    bool blk1_zero = blk[1] == 0;
    bool blk2_zero = blk[2] == 0;
    bool blk3_zero = blk[3] == 0;

    if (!blk3_zero && !blk2_zero)      pattern = SparsityPattern::P1100;
    else if (!blk3_zero && !blk1_zero) pattern = SparsityPattern::P1010;
    else if (!blk3_zero && !blk0_zero) pattern = SparsityPattern::P1001;
    else if (!blk2_zero && !blk1_zero) pattern = SparsityPattern::P0110;
    else if (!blk2_zero && !blk0_zero) pattern = SparsityPattern::P0101;
    else if (!blk1_zero && !blk0_zero) pattern = SparsityPattern::P0011;
    else throw std::runtime_error("Unexpected sparsity pattern");

    for (int j = 0; j < 4; ++j) {
        if (pattern & (1 << j)) 
            out.values.push_back(blk[j]);
    }

    out.meta.push_back(pattern);
  }

  return out;
}

// Utility for debugging. Not very useful in terms of implementation. Might as well implement it when my brain is fresh.
static std::vector<itype_t> decompress_sparse_matrix(const SparseMat sparseA) {
  uint32_t M = sparseA.rows;
  uint32_t K = sparseA.cols;
  uint32_t N = M * K;

  std::vector<itype_t> denseA(N, 0); // initialize all to zero

  const std::vector<itype_t>& vals = sparseA.values;
  const std::vector<uint8_t>& meta = sparseA.meta;

  for (uint32_t i = 0; i < N; i += 4) {
    uint8_t pattern = meta[i / 4]; // 1 entry per 4 elements
    uint8_t kept = 0;

    switch (pattern) {
        case SparsityPattern::P0011: kept = 0b0011; break;
        case SparsityPattern::P0101: kept = 0b0101; break;
        case SparsityPattern::P1001: kept = 0b1001; break;
        case SparsityPattern::P0110: kept = 0b0110; break;
        case SparsityPattern::P1010: kept = 0b1010; break;
        case SparsityPattern::P1100: kept = 0b1100; break;
        default: throw std::runtime_error("Unexpected sparsity pattern");
    }

    int k = 0;
    for (int j = 0; j < 4; ++j) {
        if (kept & (1 << j)) {
            denseA[i + j] = vals[i / 2 + k];
            k++;
        }
    }
  }

  return denseA;
}

// Another debug function.
void print_matrix_4x4(const std::vector<itype_t>& mat, int M, int K) {
    for (int i = 0; i < M; i += 4) {          // tile row
        for (int j = 0; j < K; j += 4) {      // tile column
            std::cout << "Tile starting at (" << i << "," << j << "):\n";

            for (int ti = i; ti < i + 4 && ti < M; ++ti) {
                for (int tj = j; tj < j + 4 && tj < K; ++tj) {
                    std::cout << mat[ti * K + tj] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
}

int check_matrix_8x8(const std::vector<otype_t>& dat, 
                      const std::vector<otype_t>& ref,
                      int M, 
                      int N) {
    int errors = 0;
    for (int i = 0; i < M; i += 8) { 
        for (int j = 0; j < N; j += 8) {  
            for (int ti = i; ti < i + 8 && ti < M; ++ti) {
                for (int tj = j; tj < j + 8 && tj < N; ++tj) {
                    if (!Comparator<vt::OTYPE>::compare(dat[ti * N + tj], ref[ti * N + tj], ti * N + tj, errors))
                        errors++;
                }
            }
        }
    }
    return errors;

}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = DIM_M;
uint32_t xn = DIM_N;
uint32_t xk = DIM_K;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h A_format = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

std::string last_build_options;

static void show_usage() {
  std::cout << "Vortex Sparse SGEMM TCU Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n N] [-k: K] [-s] [-h: help]" << std::endl;
  std::cout << "  -s  Enable 2:4 structured sparsity " << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:i:o:hs")) != -1) {
    switch (c) {
    case 'm':
      xm = atoi(optarg);
      break;
    case 'n':
      xn = atoi(optarg);
      break;
    case 'k':
      xk = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(A_format);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_TCU) != 0;
  if (!has_ext) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  if ((M % cfg::tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM!" << std::endl;
    return -1;
  }

  if ((N % cfg::tileN) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileN!" << std::endl;
    return -1;
  }

  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileK!" << std::endl;
    return -1;
  }

  if (cfg::tileK < 4) {
    std::cout << "You better have K be greater than 4, because the sparsity encoding doesn't work otherwise." << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "Grid Dimension: y" << N / cfg::tileN << std::endl;
  std::cout << "Grid Dimension: x" << M / cfg::tileM << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  std::cout << "NT: " << NT << std::endl;

  // set block size to warp size
  kernel_arg.grid_dim[0] = N / cfg::tileN;
  kernel_arg.grid_dim[1] = M / cfg::tileM;
  kernel_arg.block_dim[0] = NT; // warp sizeb
  kernel_arg.block_dim[1] = 1;



  // set matrix dimensions
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t) / 2, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeA / 4, VX_MEM_READ, &A_format));
  RT_CHECK(vx_mem_address(A_format, &kernel_arg.Af_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "A_format_addr=0x" << std::hex << kernel_arg.Af_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

  // generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = Comparator<vt::ITYPE>::generate();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = Comparator<vt::ITYPE>::generate();
  }

  // Sparsity pruning
  prune_dense_matrix(h_A, sizeA);
  
  // test the pruning, just to make sure
  SparseMat spMat = compress_sparse_matrix(h_A, M, K);
  std::vector<itype_t> decSpMatVec = decompress_sparse_matrix(spMat);

  for (uint32_t i = 0; i < decSpMatVec.size(); ++i) {
  if (!Comparator<vt::ITYPE>::compare(decSpMatVec[i], h_A[i], i, 0)) {
    std::cout << "SPARSE FUNCTION IS WRONG AND YOU SUCK" << std::endl;
    return 1; // Return early, no point in continuing
  }
  }

  //print_matrix_4x4(h_A, M, K); // debug
  //print_matrix_4x4(h_A, M, K);
  //print_matrix_4x4(h_B, M, K);

  // upload matrix A buffer
  {
    std::cout << "upload matrix A buffer and format" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, spMat.values.data(), 0, sizeA * sizeof(itype_t) / 2));
    RT_CHECK(vx_copy_to_dev(A_format, spMat.meta.data(), 0, sizeA / 4));
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
  }

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  std::cout << "kernel arg x " << kernel_arg.grid_dim[1] << std::endl;
  std::cout << "kernel arg y " << kernel_arg.grid_dim[0] << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

    errors = check_matrix_8x8(h_C, h_ref, M, N);
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}