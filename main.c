#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arm_neon.h>


#define RAND_MAX 0x7fffffff
#define PTR_ROW_FLAGS(ptr, row, flags) \
  (((uint64_t) & *(ptr)) + (((uint64_t)((row) + (flags)*64)) << 56))

// In Apple's Accelerate, instruction 17 is apparently always prefixed by three
// nops.
#define AMX_NOP_OP_IMM5(op, imm5)                          \
  __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" \
        :                                                  \
        : "i"(op), "i"(imm5)                               \
        : "memory")

#define AMX_OP_GPR(op, gpr)                                     \
  __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" \
        :                                                       \
        : "i"(op), "r"((uint64_t)(gpr))                         \
        : "memory")

#define AMX_LDX(gpr) AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr) AMX_OP_GPR(1, gpr)
#define AMX_STX(gpr) AMX_OP_GPR(2, gpr)
#define AMX_STY(gpr) AMX_OP_GPR(3, gpr)
#define AMX_LDZ(gpr) AMX_OP_GPR(4, gpr)
#define AMX_STZ(gpr) AMX_OP_GPR(5, gpr)
#define AMX_LDZI(gpr) AMX_OP_GPR(6, gpr)
#define AMX_STZI(gpr) AMX_OP_GPR(7, gpr)
#define AMX_EXTRX(gpr) AMX_OP_GPR(8, gpr)
#define AMX_EXTRY(gpr) AMX_OP_GPR(9, gpr)
#define AMX_FMA64(gpr) AMX_OP_GPR(10, gpr)
#define AMX_FMS64(gpr) AMX_OP_GPR(11, gpr)
#define AMX_FMA32(gpr) AMX_OP_GPR(12, gpr)
#define AMX_FMS32(gpr) AMX_OP_GPR(13, gpr)
#define AMX_MAC16(gpr) AMX_OP_GPR(14, gpr)
#define AMX_FMA16(gpr) AMX_OP_GPR(15, gpr)
#define AMX_FMS16(gpr) AMX_OP_GPR(16, gpr)
#define AMX_SET() AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(gpr) AMX_OP_GPR(18, gpr)
#define AMX_VECFP(gpr) AMX_OP_GPR(19, gpr)
#define AMX_MATINT(gpr) AMX_OP_GPR(20, gpr)
#define AMX_MATFP(gpr) AMX_OP_GPR(21, gpr)
#define AMX_GENLUT(gpr) AMX_OP_GPR(22, gpr)

#define CLOCK_START(name)  \
  printf("%s in\n", name); \
  clock_t start, end;      \
  double cpu_time_used;    \
  start = clock();

#define CLOCK_END(name)                                     \
  end = clock();                                            \
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; \
  printf("%s time: %f sec\n", name, cpu_time_used);

float *random_matrix(int rows, int cols) {
  int i;
  float *m = calloc(rows * cols, sizeof(float));
  for (i = 0; i < rows * cols; ++i) {
    m[i] = (float)rand() / RAND_MAX;
  }
  return m;
}

#define amx_z_to_x(zrow, xrow) \
  (1ull << 63) + (8ull << 11) + (1ull << 26) + ((zrow) << 20) + 64 * (xrow)


void printM(float *in, int row, int col) {
    printf("======================================\n");
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%f ", in[i * col + j]);
    }
    printf("\n");
  }
}

void printAMX_Z() {
  float debug[16];
  for (int i = 0; i < 64; i++) {
    printf("z%d:",i);
    AMX_STZ(PTR_ROW_FLAGS(debug, i, 0));
    for (int di = 0; di < 16; ++di) {
      printf("%f, ", debug[di]);
    }
    printf("\n");
  }

    printf("\n");
}

void printAMX_X() {
  float debug[16];
  for (int i = 0; i < 16; i++) {
    printf("x%d:",i);
    AMX_STX(PTR_ROW_FLAGS(debug, i, 0));
    for (int di = 0; di < 16; ++di) {
      printf("%f, ", debug[di]);
    }
    printf("\n");
  }

    printf("\n");
}

#define ZERO_Z(row)  AMX_FMA32((0ull<<63)+(7ull<<27)+(row << 20))

float ones[16] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
float zeros[16] = {0};
// static float alpha[16] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};

void gemm_nn_neon(int M, int N, int K, float ALPHA, float *ma, int lda, float *mb,
                 int ldb, float *mc, int ldc) {
  //CLOCK_START("gemm_nn_neon")

  int m, n, k;
  for (m = 0; m < M - M % 2; m += 2) {
    for (n = 0; n < N - N % 32; n += 32) {

      float32x4_t mmc0_0 = vld1q_f32(mc + (m + 0) * N + n + 0);
      float32x4_t mmc0_4 = vld1q_f32(mc + (m + 0) * N + n + 4);
      float32x4_t mmc0_8 = vld1q_f32(mc + (m + 0) * N + n + 8);
      float32x4_t mmc0_12 = vld1q_f32(mc + (m + 0) * N + n + 12);

      float32x4_t mmc0_16 = vld1q_f32(mc + (m + 0) * N + n + 16);
      float32x4_t mmc0_20 = vld1q_f32(mc + (m + 0) * N + n + 20);
      float32x4_t mmc0_24 = vld1q_f32(mc + (m + 0) * N + n + 24);
      float32x4_t mmc0_28 = vld1q_f32(mc + (m + 0) * N + n + 28);

      float32x4_t mmc1_0 = vld1q_f32(mc + (m + 1) * N + n + 0);
      float32x4_t mmc1_4 = vld1q_f32(mc + (m + 1) * N + n + 4);
      float32x4_t mmc1_8 = vld1q_f32(mc + (m + 1) * N + n + 8);
      float32x4_t mmc1_12 = vld1q_f32(mc + (m + 1) * N + n + 12);

      float32x4_t mmc1_16 = vld1q_f32(mc + (m + 1) * N + n + 16);
      float32x4_t mmc1_20 = vld1q_f32(mc + (m + 1) * N + n + 20);
      float32x4_t mmc1_24 = vld1q_f32(mc + (m + 1) * N + n + 24);
      float32x4_t mmc1_28 = vld1q_f32(mc + (m + 1) * N + n + 28);

      for (k = 0; k < K - K % 4; k += 4) {
        float ma_part_0[4] = {*(ma + (m + 0) * K + k)*ALPHA,*(ma + (m + 0) * K + k+1)*ALPHA,*(ma + (m + 0) * K + k+2)*ALPHA,*(ma + (m + 0) * K + k+3)*ALPHA};
        const float32x4_t mma0_mk = vld1q_f32(ma_part_0);
        float ma_part_1[4] = {*(ma + (m + 1) * K + k)*ALPHA,*(ma + (m + 1) * K + k+1)*ALPHA,*(ma + (m + 1) * K + k+2)*ALPHA,*(ma + (m + 1) * K + k+3)*ALPHA};
        const float32x4_t mma1_mk = vld1q_f32(ma_part_1);
        
        const float32x4_t mmb0_0 = vld1q_f32(mb + (k + 0) * N + n + 0);
        const float32x4_t mmb0_4 = vld1q_f32(mb + (k + 0) * N + n + 4);
        const float32x4_t mmb0_8 = vld1q_f32(mb + (k + 0) * N + n + 8);
        const float32x4_t mmb0_12 = vld1q_f32(mb + (k + 0) * N + n + 12);

        const float32x4_t mmb0_16 = vld1q_f32(mb + (k + 0) * N + n + 16);
        const float32x4_t mmb0_20 = vld1q_f32(mb + (k + 0) * N + n + 20);
        const float32x4_t mmb0_24 = vld1q_f32(mb + (k + 0) * N + n + 24);
        const float32x4_t mmb0_28 = vld1q_f32(mb + (k + 0) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb0_0, mma0_mk, 0);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb0_4, mma0_mk, 0);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb0_8, mma0_mk, 0);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb0_12, mma0_mk, 0);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb0_16, mma0_mk, 0);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb0_20, mma0_mk, 0);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb0_24, mma0_mk, 0);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb0_28, mma0_mk, 0);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb0_0, mma1_mk, 0);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb0_4, mma1_mk, 0);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb0_8, mma1_mk, 0);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb0_12, mma1_mk, 0);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb0_16, mma1_mk, 0);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb0_20, mma1_mk, 0);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb0_24, mma1_mk, 0);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb0_28, mma1_mk, 0);

        const float32x4_t mmb1_0 = vld1q_f32(mb + (k + 1) * N + n + 0);
        const float32x4_t mmb1_4 = vld1q_f32(mb + (k + 1) * N + n + 4);
        const float32x4_t mmb1_8 = vld1q_f32(mb + (k + 1) * N + n + 8);
        const float32x4_t mmb1_12 = vld1q_f32(mb + (k + 1) * N + n + 12);

        const float32x4_t mmb1_16 = vld1q_f32(mb + (k + 1) * N + n + 16);
        const float32x4_t mmb1_20 = vld1q_f32(mb + (k + 1) * N + n + 20);
        const float32x4_t mmb1_24 = vld1q_f32(mb + (k + 1) * N + n + 24);
        const float32x4_t mmb1_28 = vld1q_f32(mb + (k + 1) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb1_0, mma0_mk, 1);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb1_4, mma0_mk, 1);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb1_8, mma0_mk, 1);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb1_12, mma0_mk, 1);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb1_16, mma0_mk, 1);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb1_20, mma0_mk, 1);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb1_24, mma0_mk, 1);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb1_28, mma0_mk, 1);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb1_0, mma1_mk, 1);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb1_4, mma1_mk, 1);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb1_8, mma1_mk, 1);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb1_12, mma1_mk, 1);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb1_16, mma1_mk, 1);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb1_20, mma1_mk, 1);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb1_24, mma1_mk, 1);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb1_28, mma1_mk, 1);

        const float32x4_t mmb2_0 = vld1q_f32(mb + (k + 2) * N + n + 0);
        const float32x4_t mmb2_4 = vld1q_f32(mb + (k + 2) * N + n + 4);
        const float32x4_t mmb2_8 = vld1q_f32(mb + (k + 2) * N + n + 8);
        const float32x4_t mmb2_12 = vld1q_f32(mb + (k + 2) * N + n + 12);

        const float32x4_t mmb2_16 = vld1q_f32(mb + (k + 2) * N + n + 16);
        const float32x4_t mmb2_20 = vld1q_f32(mb + (k + 2) * N + n + 20);
        const float32x4_t mmb2_24 = vld1q_f32(mb + (k + 2) * N + n + 24);
        const float32x4_t mmb2_28 = vld1q_f32(mb + (k + 2) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb2_0, mma0_mk, 2);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb2_4, mma0_mk, 2);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb2_8, mma0_mk, 2);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb2_12, mma0_mk, 2);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb2_16, mma0_mk, 2);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb2_20, mma0_mk, 2);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb2_24, mma0_mk, 2);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb2_28, mma0_mk, 2);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb2_0, mma1_mk, 2);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb2_4, mma1_mk, 2);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb2_8, mma1_mk, 2);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb2_12, mma1_mk, 2);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb2_16, mma1_mk, 2);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb2_20, mma1_mk, 2);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb2_24, mma1_mk, 2);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb2_28, mma1_mk, 2);

        const float32x4_t mmb3_0 = vld1q_f32(mb + (k + 3) * N + n + 0);
        const float32x4_t mmb3_4 = vld1q_f32(mb + (k + 3) * N + n + 4);
        const float32x4_t mmb3_8 = vld1q_f32(mb + (k + 3) * N + n + 8);
        const float32x4_t mmb3_12 = vld1q_f32(mb + (k + 3) * N + n + 12);

        const float32x4_t mmb3_16 = vld1q_f32(mb + (k + 3) * N + n + 16);
        const float32x4_t mmb3_20 = vld1q_f32(mb + (k + 3) * N + n + 20);
        const float32x4_t mmb3_24 = vld1q_f32(mb + (k + 3) * N + n + 24);
        const float32x4_t mmb3_28 = vld1q_f32(mb + (k + 3) * N + n + 28);

        mmc0_0 = vmlaq_laneq_f32(mmc0_0, mmb3_0, mma0_mk, 3);
        mmc0_4 = vmlaq_laneq_f32(mmc0_4, mmb3_4, mma0_mk, 3);
        mmc0_8 = vmlaq_laneq_f32(mmc0_8, mmb3_8, mma0_mk, 3);
        mmc0_12 = vmlaq_laneq_f32(mmc0_12, mmb3_12, mma0_mk, 3);

        mmc0_16 = vmlaq_laneq_f32(mmc0_16, mmb3_16, mma0_mk, 3);
        mmc0_20 = vmlaq_laneq_f32(mmc0_20, mmb3_20, mma0_mk, 3);
        mmc0_24 = vmlaq_laneq_f32(mmc0_24, mmb3_24, mma0_mk, 3);
        mmc0_28 = vmlaq_laneq_f32(mmc0_28, mmb3_28, mma0_mk, 3);

        mmc1_0 = vmlaq_laneq_f32(mmc1_0, mmb3_0, mma1_mk, 3);
        mmc1_4 = vmlaq_laneq_f32(mmc1_4, mmb3_4, mma1_mk, 3);
        mmc1_8 = vmlaq_laneq_f32(mmc1_8, mmb3_8, mma1_mk, 3);
        mmc1_12 = vmlaq_laneq_f32(mmc1_12, mmb3_12, mma1_mk, 3);

        mmc1_16 = vmlaq_laneq_f32(mmc1_16, mmb3_16, mma1_mk, 3);
        mmc1_20 = vmlaq_laneq_f32(mmc1_20, mmb3_20, mma1_mk, 3);
        mmc1_24 = vmlaq_laneq_f32(mmc1_24, mmb3_24, mma1_mk, 3);
        mmc1_28 = vmlaq_laneq_f32(mmc1_28, mmb3_28, mma1_mk, 3);
      }

      vst1q_f32(mc + (m + 0) * N + n + 0, mmc0_0);
      vst1q_f32(mc + (m + 0) * N + n + 4, mmc0_4);
      vst1q_f32(mc + (m + 0) * N + n + 8, mmc0_8);
      vst1q_f32(mc + (m + 0) * N + n + 12, mmc0_12);

      vst1q_f32(mc + (m + 0) * N + n + 16, mmc0_16);
      vst1q_f32(mc + (m + 0) * N + n + 20, mmc0_20);
      vst1q_f32(mc + (m + 0) * N + n + 24, mmc0_24);
      vst1q_f32(mc + (m + 0) * N + n + 28, mmc0_28);

      vst1q_f32(mc + (m + 1) * N + n + 0, mmc1_0);
      vst1q_f32(mc + (m + 1) * N + n + 4, mmc1_4);
      vst1q_f32(mc + (m + 1) * N + n + 8, mmc1_8);
      vst1q_f32(mc + (m + 1) * N + n + 12, mmc1_12);

      vst1q_f32(mc + (m + 1) * N + n + 16, mmc1_16);
      vst1q_f32(mc + (m + 1) * N + n + 20, mmc1_20);
      vst1q_f32(mc + (m + 1) * N + n + 24, mmc1_24);
      vst1q_f32(mc + (m + 1) * N + n + 28, mmc1_28);
    }
  }

   for (; m < M; ++m) {
    for (n = 0; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        register float A_PART = ALPHA * ma[m * lda + k];
        mc[m * ldc + n] += A_PART * mb[k * ldb + n];
      }
    }
  }
  //CLOCK_END("gemm_nn_neon")
}



#define ELELIST(a) \
  { a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a }

#define A_LIST(p)                                                              \
  const float amk##p[16] = {A[m * K + k + p] * ALPHA,    A[(m + 1) * K + k+ p] * ALPHA,   \
                    A[(m + 2) * K + k+ p] * ALPHA,  A[(m + 3) * K + k+ p] * ALPHA,   \
                    A[(m + 4) * K + k+ p] * ALPHA,  A[(m + 5) * K + k+ p] * ALPHA,   \
                    A[(m + 6) * K + k+ p] * ALPHA,  A[(m + 7) * K + k+ p] * ALPHA,   \
                    A[(m + 8) * K + k+ p] * ALPHA,  A[(m + 9) * K + k+ p] * ALPHA,   \
                    A[(m + 10) * K + k+ p] * ALPHA, A[(m + 11) * K + k+ p] * ALPHA,  \
                    A[(m + 12) * K + k+ p] * ALPHA, A[(m + 13) * K + k+ p] * ALPHA,  \
                    A[(m + 14) * K + k+ p] * ALPHA, A[(m + 15) * K + k+ p] * ALPHA}
void gemm_nn_amx5(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
                 int ldb, float *C, int ldc) {
  //CLOCK_START("gemm_nn_amx5")
  AMX_SET();
  int m, n, k;
  /**
   C0 += A0[0] * B0
   C1 += A1[0] * B0
   C2 += A2[0] * B0
   ...
   C15 += A15[0] * B0

   C0 += A0[1] * B1
   C1 += A1[1] * B1
   C2 += A2[1] * B1
   ...
   C15 += A15[1] * B1
   .
   .
   .
   C0 += A0[15] * B15
   C1 += A1[15] * B15
   C2 += A2[15] * B15
   ...
   C15 += A15[15] * B15
   **/
  

  for (m = 0; m < M - M % 16; m += 16) {
    for (n = 0; n < N - N % 16; n += 16) {
      float *rC0 = C + (m + 0) * N + n;
      float *rC1 = C + (m + 1) * N + n;
      float *rC2 = C + (m + 2) * N + n;
      float *rC3 = C + (m + 3) * N + n;
      float *rC4 = C + (m + 4) * N + n;
      float *rC5 = C + (m + 5) * N + n;
      float *rC6 = C + (m + 6) * N + n;
      float *rC7 = C + (m + 7) * N + n;
      float *rC8 = C + (m + 8) * N + n;
      float *rC9 = C + (m + 9) * N + n;
      float *rC10 = C + (m + 10) * N + n;
      float *rC11 = C + (m + 11) * N + n;
      float *rC12 = C + (m + 12) * N + n;
      float *rC13 = C + (m + 13) * N + n;
      float *rC14 = C + (m + 14) * N + n;
      float *rC15 = C + (m + 15) * N + n;

      AMX_CLR();
      AMX_SET();
      for (k = 0; k < K - K % 16; k += 16) {
        
        A_LIST(0);
        A_LIST(1);
        A_LIST(2);
        A_LIST(3);
        A_LIST(4);
        A_LIST(5);
        A_LIST(6);
        A_LIST(7);
        A_LIST(8);
        A_LIST(9);
        A_LIST(10);
        A_LIST(11);
        A_LIST(12);
        A_LIST(13);
        A_LIST(14);
        A_LIST(15);

        const float *rB0 = B + (k + 0) * N + n;
        const float *rB1 = B + (k + 1) * N + n;
        const float *rB2 = B + (k + 2) * N + n;
        const float *rB3 = B + (k + 3) * N + n;
        const float *rB4 = B + (k + 4) * N + n;
        const float *rB5 = B + (k + 5) * N + n;
        const float *rB6 = B + (k + 6) * N + n;
        const float *rB7 = B + (k + 7) * N + n;
        const float *rB8 = B + (k + 8) * N + n;
        const float *rB9 = B + (k + 9) * N + n;
        const float *rB10 = B + (k + 10) * N + n;
        const float *rB11 = B + (k + 11) * N + n;
        const float *rB12 = B + (k + 12) * N + n;
        const float *rB13 = B + (k + 13) * N + n;
        const float *rB14 = B + (k + 14) * N + n;
        const float *rB15 = B + (k + 15) * N + n;

        AMX_LDY(PTR_ROW_FLAGS(amk0, 0, 0)); 
        AMX_LDX(PTR_ROW_FLAGS(rB0, 0, 0));       
        AMX_MATFP((4ull << 42));  //每一个A0[i] 点乘 B0 然后累加到C0-C15
        AMX_LDY(PTR_ROW_FLAGS(amk1, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB1, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk2, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB2, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk3, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB3, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk4, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB4, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk5, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB5, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk6, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB6, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk7, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB7, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk8, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB8, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk9, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB9, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk10, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB10, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk11, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB11, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk12, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB12, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk13, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB13, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk14, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB14, 0, 0));  
        AMX_MATFP((4ull << 42));      
        AMX_LDY(PTR_ROW_FLAGS(amk15, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(rB15, 0, 0));  
        AMX_MATFP((4ull << 42));
      }
      // C0-C15对应 z0 z4 z8 ... z60
      AMX_STZ(PTR_ROW_FLAGS(rC0, 0, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC1, 4, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC2, 8, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC3, 12, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC4, 16, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC5, 20, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC6, 24, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC7, 28, 0));

      AMX_STZ(PTR_ROW_FLAGS(rC8, 32, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC9, 36, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC10, 40, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC11, 44, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC12, 48, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC13, 52, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC14, 56, 0));
      AMX_STZ(PTR_ROW_FLAGS(rC15, 60, 0));
    }
  }
  
  for (; m < M; ++m) {
    for (n = 0; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        register float A_PART = ALPHA * A[m * lda + k];
        C[m * ldc + n] += A_PART * B[k * ldb + n];
      }
    }
  }

  AMX_CLR();
  //CLOCK_END("gemm_nn_amx5")
}


void gemm_nn_amx4(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
                 int ldb, float *C, int ldc) {
  //CLOCK_START("gemm_nn_amx4")
  int flag = 1;
  AMX_SET();
  static float zero[16] = {0};
  int m, n, k;
  float alpha[16] = {ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA,
                     ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA};
#define AMK(p)                                                                 \
  float amk##p[16] = {                                                         \
    A[(m + 0+16*p) * K + k ],                                                    \
    A[(m + 1+16*p) * K + k ],                                                    \
    A[(m + 2+16*p) * K + k ],                                                    \
    A[(m + 3+16*p) * K + k ],                                                    \
    A[(m + 4+16*p) * K + k ],                                                    \
    A[(m + 5+16*p) * K + k ],                                                    \
    A[(m + 6+16*p) * K + k ],                                                    \
    A[(m + 7+16*p) * K + k ],                                                    \
    A[(m + 8+16*p) * K + k ],                                                    \
    A[(m + 9+16*p) * K + k ],                                                    \
    A[(m + 10+16*p) * K + k ],                                                   \
    A[(m + 11+16*p) * K + k ],                                                   \
    A[(m + 12+16*p) * K + k ],                                                   \
    A[(m + 13+16*p) * K + k ],                                                   \
    A[(m + 14+16*p) * K + k ],                                                   \
    A[(m + 15+16*p) * K + k ]                                                    \
  }
  for (m = 0; m < M - M % 32; m += 32) {
    for (n = 0; n < N - N % 16; n += 16) {
      AMX_CLR();
      AMX_SET();

      for (k = 0; k < K ; ++k) {
        AMK(0);
        AMK(1);
        // AMK(2);

        ZERO_Z(0);ZERO_Z(4);ZERO_Z(8);ZERO_Z(12);ZERO_Z(16);ZERO_Z(20);ZERO_Z(24);ZERO_Z(28);
        ZERO_Z(32);ZERO_Z(36);ZERO_Z(40);ZERO_Z(44);ZERO_Z(48);ZERO_Z(52);ZERO_Z(56);ZERO_Z(60); 

       // 1 2 3,5 6 7,9 10 11,13 14 15, 17 18 19, 21 
       // 22 23, 25 26 27, 29 30 31, 33 34 35, 37 38 39 41 42 
       //43 45 46 47 49 50 51 53 54 55 57 58 59 61 62 63
        AMX_LDY(PTR_ROW_FLAGS(alpha, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(amk0, 0, 0));
        AMX_FMA32((0ull << 20)); //z0 z4 ... is A*APLHA

        // 26 must be1      +  f32     f32         + z colum
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) );                  // x0 is [amk0 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (4ull<<20) +64*1); // x1 is [amk1 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (8ull<<20) +64*2);  // x2 is [amk2 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (12ull<<20)+64*3); // x3 is [amk3 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (16ull<<20)+64*4); // x4 is [amk4 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (20ull<<20)+64*5); // x5 is [amk5 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (24ull<<20)+64*6); // x6 is [amk6 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (28ull<<20)+64*7); // x7 is [amk7 x16]

        AMX_LDY(PTR_ROW_FLAGS(B + k * N + n, 0, 0));  // load B to Y0 (16 float)

        // vectormode + zrow + xoffset
        AMX_FMA32((1ull<<63)+(1ull<<20)  );               
        AMX_FMA32((1ull<<63)+(2ull<<20) +(64ull<<10));    
        AMX_FMA32((1ull<<63)+(3ull<<20) +(128ull<<10));  
        AMX_FMA32((1ull<<63)+(5ull<<20) +(192ull<<10));  
        AMX_FMA32((1ull<<63)+(6ull<<20) +(256ull<<10));  
        AMX_FMA32((1ull<<63)+(7ull<<20) +(320ull<<10));   
        AMX_FMA32((1ull<<63)+(9ull<<20) +(384ull<<10));   
        AMX_FMA32((1ull<<63)+(10ull<<20) +(448ull<<10));     

        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (32ull<<20));     // x0 is [amk8 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (36ull<<20) +64*1); // x1 is [amk9 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (40ull<<20) +64*2);  // x2 is [amk10 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (44ull<<20)+64*3); // x3 is [amk11 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (48ull<<20)+64*4); // x4 is [amk12 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (52ull<<20)+64*5); // x5 is [amk13 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (56ull<<20)+64*6); // x6 is [amk14 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (60ull<<20)+64*7); // x7 is [amk15 x16]

        AMX_FMA32((1ull<<63)+(11ull<<20));                
        AMX_FMA32((1ull<<63)+(13ull<<20) +(64ull<<10));   
        AMX_FMA32((1ull<<63)+(14ull<<20) +(128ull<<10));   
        AMX_FMA32((1ull<<63)+(15ull<<20) +(192ull<<10));   
        AMX_FMA32((1ull<<63)+(17ull<<20) +(256ull<<10));    
        AMX_FMA32((1ull<<63)+(18ull<<20) +(320ull<<10));   
        AMX_FMA32((1ull<<63)+(19ull<<20) +(384ull<<10));    
        AMX_FMA32((1ull<<63)+(21ull<<20) +(448ull<<10));      
/////////////////////////////////////////////////////////////////////////////////////////////////

        ZERO_Z(0);ZERO_Z(4);ZERO_Z(8);ZERO_Z(12);ZERO_Z(16);ZERO_Z(20);ZERO_Z(24);ZERO_Z(28);
        ZERO_Z(32);ZERO_Z(36);ZERO_Z(40);ZERO_Z(44);ZERO_Z(48);ZERO_Z(52);ZERO_Z(56);ZERO_Z(60); 

       // 1 2 3,5 6 7,9 10 11,13 14 15, 17 18 19, 21 
       // 22 23, 25 26 27, 29 30 31, 33 34 35, 37 38 39 41 42 
       //43 45 46 47 49 50 51 53 54 55 57 58 59 61 62 63
        AMX_LDY(PTR_ROW_FLAGS(alpha, 0, 0));
        AMX_LDX(PTR_ROW_FLAGS(amk1, 0, 0));
        AMX_FMA32((0ull << 20)); //z0 z4 ... is A*APLHA

        // 26 must be1      +  f32     f32         + z colum
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) );                  // x0 is [amk0 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (4ull<<20) +64*1); // x1 is [amk1 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (8ull<<20) +64*2);  // x2 is [amk2 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (12ull<<20)+64*3); // x3 is [amk3 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (16ull<<20)+64*4); // x4 is [amk4 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (20ull<<20)+64*5); // x5 is [amk5 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (24ull<<20)+64*6); // x6 is [amk6 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (28ull<<20)+64*7); // x7 is [amk7 x16]

        AMX_LDY(PTR_ROW_FLAGS(B + k * N + n, 0, 0));  // load B to Y0 (16 float)

        // vectormode + zrow + xoffset
        AMX_FMA32((1ull<<63)+(22ull<<20)  );               
        AMX_FMA32((1ull<<63)+(23ull<<20) +(64ull<<10));    
        AMX_FMA32((1ull<<63)+(25ull<<20) +(128ull<<10));  
        AMX_FMA32((1ull<<63)+(26ull<<20) +(192ull<<10));  
        AMX_FMA32((1ull<<63)+(27ull<<20) +(256ull<<10));  
        AMX_FMA32((1ull<<63)+(29ull<<20) +(320ull<<10));   
        AMX_FMA32((1ull<<63)+(30ull<<20) +(384ull<<10));   
        AMX_FMA32((1ull<<63)+(31ull<<20) +(448ull<<10));     

        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (32ull<<20));     // x0 is [amk8 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (36ull<<20) +64*1); // x1 is [amk9 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (40ull<<20) +64*2);  // x2 is [amk10 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (44ull<<20)+64*3); // x3 is [amk11 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (48ull<<20)+64*4); // x4 is [amk12 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (52ull<<20)+64*5); // x5 is [amk13 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (56ull<<20)+64*6); // x6 is [amk14 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (60ull<<20)+64*7); // x7 is [amk15 x16]

        AMX_FMA32((1ull<<63)+(33ull<<20));                
        AMX_FMA32((1ull<<63)+(34ull<<20) +(64ull<<10));   
        AMX_FMA32((1ull<<63)+(35ull<<20) +(128ull<<10));   
        AMX_FMA32((1ull<<63)+(37ull<<20) +(192ull<<10));   
        AMX_FMA32((1ull<<63)+(38ull<<20) +(256ull<<10));    
        AMX_FMA32((1ull<<63)+(39ull<<20) +(320ull<<10));   
        AMX_FMA32((1ull<<63)+(41ull<<20) +(384ull<<10));    
        AMX_FMA32((1ull<<63)+(42ull<<20) +(448ull<<10));      
/////////////////////////////////////////////////////////////////////////////////////////////////


        // ZERO_Z(0);ZERO_Z(4);ZERO_Z(8);ZERO_Z(12);ZERO_Z(16);ZERO_Z(20);ZERO_Z(24);ZERO_Z(28);
        // ZERO_Z(32);ZERO_Z(36);ZERO_Z(40);ZERO_Z(44);ZERO_Z(48);ZERO_Z(52);ZERO_Z(56);ZERO_Z(60); 

        // AMX_LDY(PTR_ROW_FLAGS(alpha, 0, 0));
        // AMX_LDX(PTR_ROW_FLAGS(amk2, 0, 0));
        // AMX_FMA32((0ull << 20)); //z0 z4 ... is A*APLHA

        // // 26 must be1      +  f32     f32         + z colum
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) );                  // x0 is [amk0 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (4ull<<20) +64*1); // x1 is [amk1 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (8ull<<20) +64*2);  // x2 is [amk2 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (12ull<<20)+64*3); // x3 is [amk3 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (16ull<<20)+64*4); // x4 is [amk4 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (20ull<<20)+64*5); // x5 is [amk5 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (24ull<<20)+64*6); // x6 is [amk6 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (28ull<<20)+64*7); // x7 is [amk7 x16]

        // AMX_LDY(PTR_ROW_FLAGS(B + k * N + n, 0, 0));  // load B to Y0 (16 float)
        // // vectormode + zrow + xoffset
        // AMX_FMA32((1ull<<63)+(43ull<<20)  );                   //z43 += x0*y0
        // AMX_FMA32((1ull<<63)+(45ull<<20) +(64ull<<10));      // z45 += x1*y0
        // AMX_FMA32((1ull<<63)+(46ull<<20) +(128ull<<10));     // z46 += x2*y0
        // AMX_FMA32((1ull<<63)+(47ull<<20) +(192ull<<10));    // z47 += x3*y0
        // AMX_FMA32((1ull<<63)+(49ull<<20) +(256ull<<10));     // z49 += x4*y0
        // AMX_FMA32((1ull<<63)+(50ull<<20) +(320ull<<10));     // z50 += x5*y0
        // AMX_FMA32((1ull<<63)+(51ull<<20) +(384ull<<10));    // z51 += x6*y0
        // AMX_FMA32((1ull<<63)+(53ull<<20) +(448ull<<10));      // z53 += x7*y0

        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (32ull<<20));     // x0 is [amk8 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (36ull<<20) +64*1); // x1 is [amk9 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (40ull<<20) +64*2);  // x2 is [amk10 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (44ull<<20)+64*3); // x3 is [amk11 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (48ull<<20)+64*4); // x4 is [amk12 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (52ull<<20)+64*5); // x5 is [amk13 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (56ull<<20)+64*6); // x6 is [amk14 x16]
        // AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (60ull<<20)+64*7); // x7 is [amk15 x16]

        // AMX_FMA32((1ull<<63)+(54ull<<20));                   //z54 += x0*y0
        // AMX_FMA32((1ull<<63)+(55ull<<20) +(64ull<<10));      // z55 += x1*y0
        // AMX_FMA32((1ull<<63)+(57ull<<20) +(128ull<<10));     // z57 += x2*y0
        // AMX_FMA32((1ull<<63)+(58ull<<20) +(192ull<<10));    // z58 += x3*y0
        // AMX_FMA32((1ull<<63)+(59ull<<20) +(256ull<<10));     // z59 += x4*y0
        // AMX_FMA32((1ull<<63)+(61ull<<20) +(320ull<<10));     // z61+= x5*y0
        // AMX_FMA32((1ull<<63)+(62ull<<20) +(384ull<<10));    // z62+= x6*y0
        // AMX_FMA32((1ull<<63)+(63ull<<20) +(448ull<<10));      // z63 += x7*y0
/////////////////////////////////////////////////////////////////////////////////////////////////


      }

       // 1 2 3,5 6 7,9 10 11,13 14 15, 17 18 19, 21 
       // 22 23, 25 26 27, 29 30 31, 33 34 35, 37 38 39 41 42 
       //43 45 46 47 49 50 51 53 54 55 57 58 59 61 62 63

      AMX_STZ(PTR_ROW_FLAGS(C + m * N + n, 1, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 1)* N + n, 2, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 2) * N + n, 3, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 3) * N + n, 5, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 4) * N + n, 6, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 5) * N + n, 7, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 6) * N + n, 9, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 7) * N + n, 10, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 8) * N + n, 11, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 9) * N + n, 13, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 10) * N + n, 14, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 11) * N + n, 15, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 12) * N + n, 17, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 13) * N + n, 18, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 14) * N + n, 19, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 15) * N + n, 21, 0));
      
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 16) * N + n, 22, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 17) * N + n, 23, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 18) * N + n, 25, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 19) * N + n, 26, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 20) * N + n, 27, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 21) * N + n, 29, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 22) * N + n, 30, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 23) * N + n, 31, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 24) * N + n, 33, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 25) * N + n, 34, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 26) * N + n, 35, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 27) * N + n, 37, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 28) * N + n, 38, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 29) * N + n, 39, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 30) * N + n, 41, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 31) * N + n, 42, 0));

      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 32) * N + n, 43, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 33) * N + n, 45, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 34) * N + n, 46, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 35) * N + n, 47, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 36) * N + n, 49, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 37) * N + n, 50, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 38) * N + n, 51, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 39) * N + n, 53, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 40) * N + n, 54, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 41) * N + n, 55, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 42) * N + n, 57, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 43) * N + n, 58, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 44) * N + n, 59, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 45) * N + n, 61, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 46) * N + n, 62, 0));
      // AMX_STZ(PTR_ROW_FLAGS(C + (m + 47) * N + n, 63, 0));
      
    }
  }

  for (; m < M; ++m) {
    for (n = 0; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        register float A_PART = ALPHA * A[m * lda + k];
        C[m * ldc + n] += A_PART * B[k * ldb + n];
      }
    }
  }

  AMX_CLR();
  //CLOCK_END("gemm_nn_amx4")
}


void gemm_nn_amx3(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
                 int ldb, float *C, int ldc) {
  //CLOCK_START("gemm_nn_amx3")
  int flag = 1;
  AMX_SET();
  // static float zero[16] = {0};
  int m, n, k;
  float alpha[16] = {ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA,
                     ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA};
  for (m = 0; m < M - M % 16; m += 16) {
    for (n = 0; n < N - N % 16; n += 16) {
      AMX_CLR();
      AMX_SET();

      for (k = 0; k < K; ++k) {
        float debug[16];

        float amk[16] = {
            A[(m + 0) * K + k],  A[(m + 1) * K + k],  A[(m + 2) * K + k],
            A[(m + 3) * K + k],  A[(m + 4) * K + k],  A[(m + 5) * K + k],
            A[(m + 6) * K + k],  A[(m + 7) * K + k],  A[(m + 8) * K + k],
            A[(m + 9) * K + k],  A[(m + 10) * K + k], A[(m + 11) * K + k],
            A[(m + 12) * K + k], A[(m + 13) * K + k], A[(m + 14) * K + k],
            A[(m + 15) * K + k]};
            

        AMX_LDY(PTR_ROW_FLAGS(alpha, 0, 0));
        ZERO_Z(0);
        ZERO_Z(4);
        ZERO_Z(8);
        ZERO_Z(12);
        ZERO_Z(16);
        ZERO_Z(20);
        ZERO_Z(24);
        ZERO_Z(28);
        ZERO_Z(32);
        ZERO_Z(36);
        ZERO_Z(40);
        ZERO_Z(44);
        ZERO_Z(48);
        ZERO_Z(52);
        ZERO_Z(56);
        ZERO_Z(60);
      // printAMX_Z();
        AMX_LDX(PTR_ROW_FLAGS(amk, 0, 0));
        AMX_FMA32((0ull << 20)); //z0 z4 ... is A*APLHA
      // printAMX_Z();
        // printf("alpha:");
        // AMX_STY(PTR_ROW_FLAGS(debug, 0, 0));
        // for (int di = 0; di < 16; ++di) {
        //   printf("%f, ", debug[di]);
        // }
        // printf("\n");

        // printf("amk[16]:");
        // for (int di = 0; di < 16; ++di) {
        //   printf("%f, ", amk[di]);
        // }
        // printf("\n");

        // 26 must be1      +  f32     f32         + z colum
         AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) );                  // x0 is [amk0 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (4ull<<20) +64*1); // x1 is [amk1 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (8ull<<20) +64*2);  // x2 is [amk2 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (12ull<<20)+64*3); // x3 is [amk3 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (16ull<<20)+64*4); // x4 is [amk4 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (20ull<<20)+64*5); // x5 is [amk5 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (24ull<<20)+64*6); // x6 is [amk6 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (28ull<<20)+64*7); // x7 is [amk7 x16]

    // 43   ,45 46 47 , 49 50 51 , 53     54 55 , 57 58 59  ,61 62 63  used to store result
        AMX_LDY(PTR_ROW_FLAGS(B + k * N + n, 0, 0));  // load B to Y0 (16 float)

        // vectormode + zrow + xoffset
        AMX_FMA32((1ull<<63)+(43ull<<20)  );                   //z43 += x0*y0
        AMX_FMA32((1ull<<63)+(45ull<<20) +(64ull<<10));      // z45 += x1*y0
        AMX_FMA32((1ull<<63)+(46ull<<20) +(128ull<<10));     // z46 += x2*y0
        AMX_FMA32((1ull<<63)+(47ull<<20) +(192ull<<10));    // z47 += x3*y0
        AMX_FMA32((1ull<<63)+(49ull<<20) +(256ull<<10));     // z49 += x4*y0
        AMX_FMA32((1ull<<63)+(50ull<<20) +(320ull<<10));     // z50 += x5*y0
        AMX_FMA32((1ull<<63)+(51ull<<20) +(384ull<<10));    // z51 += x6*y0
        AMX_FMA32((1ull<<63)+(53ull<<20) +(448ull<<10));      // z53 += x7*y0

        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (32ull<<20));     // x0 is [amk8 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (36ull<<20) +64*1); // x1 is [amk9 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (40ull<<20) +64*2);  // x2 is [amk10 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (44ull<<20)+64*3); // x3 is [amk11 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (48ull<<20)+64*4); // x4 is [amk12 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (52ull<<20)+64*5); // x5 is [amk13 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (56ull<<20)+64*6); // x6 is [amk14 x16]
        AMX_EXTRY((1ull<<26)+(8ull<<11)+(1ull<<63) + (60ull<<20)+64*7); // x7 is [amk15 x16]

        AMX_FMA32((1ull<<63)+(54ull<<20));                   //z54 += x0*y0
        AMX_FMA32((1ull<<63)+(55ull<<20) +(64ull<<10));      // z55 += x1*y0
        AMX_FMA32((1ull<<63)+(57ull<<20) +(128ull<<10));     // z57 += x2*y0
        AMX_FMA32((1ull<<63)+(58ull<<20) +(192ull<<10));    // z58 += x3*y0
        AMX_FMA32((1ull<<63)+(59ull<<20) +(256ull<<10));     // z59 += x4*y0
        AMX_FMA32((1ull<<63)+(61ull<<20) +(320ull<<10));     // z61+= x5*y0
        AMX_FMA32((1ull<<63)+(62ull<<20) +(384ull<<10));    // z62+= x6*y0
        AMX_FMA32((1ull<<63)+(63ull<<20) +(448ull<<10));      // z63 += x7*y0

      }

      // 43   ,45 46 47 , 49 50 51 , 53     54 55 , 57 58 59  ,61 62 63  used to
      // store result
      AMX_STZ(PTR_ROW_FLAGS(C + m * N + n, 43, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 1) * N + n, 45, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 2) * N + n, 46, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 3) * N + n, 47, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 4) * N + n, 49, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 5) * N + n, 50, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 6) * N + n, 51, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 7) * N + n, 53, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 8) * N + n, 54, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 9) * N + n, 55, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 10) * N + n, 57, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 11) * N + n, 58, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 12) * N + n, 59, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 13) * N + n, 61, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 14) * N + n, 62, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 15) * N + n, 63, 0));
    }
  }

  for (; m < M; ++m) {
    for (; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        register float A_PART = ALPHA * A[m * lda + k];
        C[m * ldc + n] += A_PART * B[k * ldb + n];
      }
    }
  }

  AMX_CLR();
  //CLOCK_END("gemm_nn_amx3")
}

void gemm_nn_amx2(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
                  int ldb, float *C, int ldc) {
    //CLOCK_START("gemm_nn_amx2")
int flag =1;
  static float zero[16] = {0};
  static float alpha[16] = {0};
  AMX_SET();
  int m, n, k;
  for (m = 0; m < M - M % 16; m += 16) {
    for (n = 0; n < N - N % 16; n += 16) {
      AMX_LDZ(PTR_ROW_FLAGS(zero, 0, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 1, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 2, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 3, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 4, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 5, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 6, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 7, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 8, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 9, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 10, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 11, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 12, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 13, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 14, 0));
      AMX_LDZ(PTR_ROW_FLAGS(zero, 15, 0));
      for (k = 0; k < K; ++k) {
        // generate A_PART vector
        float amk0 = A[m * K + k] * ALPHA;
        float amk1 = A[(m + 1) * K + k] * ALPHA;
        float amk2 = A[(m + 2) * K + k] * ALPHA;
        float amk3 = A[(m + 3) * K + k] * ALPHA;
        float amk4 = A[(m + 4) * K + k] * ALPHA;
        float amk5 = A[(m + 5) * K + k] * ALPHA;
        float amk6 = A[(m + 6) * K + k] * ALPHA;
        float amk7 = A[(m + 7) * K + k] * ALPHA;
        float amk8 = A[(m + 8) * K + k] * ALPHA;
        float amk9 = A[(m + 9) * K + k] * ALPHA;
        float amk10 = A[(m + 10) * K + k] * ALPHA;
        float amk11 = A[(m + 11) * K + k] * ALPHA;
        float amk12 = A[(m + 12) * K + k] * ALPHA;
        float amk13 = A[(m + 13) * K + k] * ALPHA;
        float amk14 = A[(m + 14) * K + k] * ALPHA;
        float amk15 = A[(m + 15) * K + k] * ALPHA;
#define AMK_DUP(a) \
  { a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a }
        float AMK0[16] = AMK_DUP(amk0);
        float AMK1[16] = AMK_DUP(amk1);
        float AMK2[16] = AMK_DUP(amk2);
        float AMK3[16] = AMK_DUP(amk3);
        float AMK4[16] = AMK_DUP(amk4);
        float AMK5[16] = AMK_DUP(amk5);
        float AMK6[16] = AMK_DUP(amk6);
        float AMK7[16] = AMK_DUP(amk7);
        float AMK8[16] = AMK_DUP(amk8);
        float AMK9[16] = AMK_DUP(amk9);
        float AMK10[16] = AMK_DUP(amk10);
        float AMK11[16] = AMK_DUP(amk11);
        float AMK12[16] = AMK_DUP(amk12);
        float AMK13[16] = AMK_DUP(amk13);
        float AMK14[16] = AMK_DUP(amk14);
        float AMK15[16] = AMK_DUP(amk15);

        AMX_LDY(PTR_ROW_FLAGS(B + k * N + n, 0, 0));  // load B to Y0 (16 float)

        AMX_LDX(PTR_ROW_FLAGS(AMK0, 0, 0));

        AMX_FMA32((1ull << 63));
        AMX_LDX(PTR_ROW_FLAGS(AMK1, 0, 0));
        AMX_FMA32((1ull << 63) + (1ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK2, 0, 0));
        AMX_FMA32((1ull << 63) + (2ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK3, 0, 0));
        AMX_FMA32((1ull << 63) + (3ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK4, 0, 0));
        AMX_FMA32((1ull << 63) + (4ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK5, 0, 0));
        AMX_FMA32((1ull << 63) + (5ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK6, 0, 0));
        AMX_FMA32((1ull << 63) + (6ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK7, 0, 0));
        AMX_FMA32((1ull << 63) + (7ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK8, 0, 0));
        AMX_FMA32((1ull << 63) + (8ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK9, 0, 0));
        AMX_FMA32((1ull << 63) + (9ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK10, 0, 0));
        AMX_FMA32((1ull << 63) + (10ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK11, 0, 0));
        AMX_FMA32((1ull << 63) + (11ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK12, 0, 0));
        AMX_FMA32((1ull << 63) + (12ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK13, 0, 0));
        AMX_FMA32((1ull << 63) + (13ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK14, 0, 0));
        AMX_FMA32((1ull << 63) + (14ull << 20));
        AMX_LDX(PTR_ROW_FLAGS(AMK15, 0, 0));
        AMX_FMA32((1ull << 63) + (15ull << 20));

      }
      
      AMX_STZ(PTR_ROW_FLAGS(C + m * N + n, 0, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 1) * N + n, 1, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 2) * N + n, 2, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 3) * N + n, 3, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 4) * N + n, 4, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 5) * N + n, 5, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 6) * N + n, 6, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 7) * N + n, 7, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 8) * N + n, 8, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 9) * N + n, 9, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 10) * N + n, 10, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 11) * N + n, 11, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 12) * N + n, 12, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 13) * N + n, 13, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 14) * N + n, 14, 0));
      AMX_STZ(PTR_ROW_FLAGS(C + (m + 15) * N + n, 15, 0));
    }
  }

  for (; m < M; ++m) {
    for (; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        register float A_PART = ALPHA * A[m * lda + k];
        C[m * ldc + n] += A_PART * B[k * ldb + n];
      }
    }
  }

  AMX_CLR();
    //CLOCK_END("gemm_nn_amx2")
}

void gemm_nn_amx1(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
                 int ldb, float *C, int ldc) {
  //CLOCK_START("gemm_nn_amx1")

  AMX_SET();
  float zero[16] = {0};
  int m, n, k;
  float alpha[16] = {ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA,
                     ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA, ALPHA};
  for (m = 0; m < M; ++m) {
    for (n = 0; n < N - N % 16; n += 16) {
      AMX_LDZ(PTR_ROW_FLAGS(zero, 0, 0));
      for (k = 0; k < K; ++k) {
        //generate A_PART vector
        float amk = A[m * K + k]*ALPHA;
        float Amk[16]= {amk,amk,amk,amk,amk,amk,amk,amk,amk,amk,amk,amk,amk,amk,amk,amk};
        AMX_LDX(PTR_ROW_FLAGS(Amk, 0, 0));
        AMX_LDY(PTR_ROW_FLAGS(B + k * N + n, 0, 0));  // load B to Y0

        AMX_FMA32(0x800000000000000);                // Z0 += X0*Y0
      }
      AMX_STZ(PTR_ROW_FLAGS(C + m * N + n, 0, 0));
    }

    for (; n < N; ++n) {
      for (k = 0; k < K; ++k) {
        register float A_PART = ALPHA * A[m * lda + k];
        C[m * ldc + n] += A_PART * B[k * ldb + n];
      }
    }
  }
  AMX_CLR();
  //CLOCK_END("gemm_nn_amx1")
}

void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
             int ldb, float *C, int ldc) {
  //CLOCK_START("gemm_nn")
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      register float A_PART = ALPHA * A[i * lda + k];
      for (j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }

//CLOCK_END("gemm_nn")
}

int main(int argc, char **argv) {
  int EPOCH = 100;
  for (int SIZE = 16; SIZE < 1024; SIZE += 64) {
    int M = SIZE;
    int K = SIZE;
    int N = SIZE;
    float *C1 = calloc(M * N, sizeof(float));
    float *A = random_matrix(M, K);
    float *B = random_matrix(K, N);
    double total_time1 = 0.0;
    double total_time2 = 0.0;
    double total_time3 = 0.0;
    for (int cnt = 0; cnt < EPOCH; cnt++) {
      clock_t start, end;
      double cpu_time_used;
      start = clock();
      gemm_nn(M, N, K, 1, A, K, B, N, C1, N);
      end = clock();
      total_time1 += ((double)(end - start))*1000  / CLOCKS_PER_SEC;
    }

    for (int cnt = 0; cnt < EPOCH; cnt++) {
      clock_t start, end;
      double cpu_time_used;
      start = clock();
      gemm_nn_amx5(M, N, K, 1, A, K, B, N, C1, N);
      end = clock();
      total_time2 += ((double)(end - start))*1000  / CLOCKS_PER_SEC;
    }

    for (int cnt = 0; cnt < EPOCH; cnt++) {
      clock_t start, end;
      double cpu_time_used;
      start = clock();
      gemm_nn_neon(M, N, K, 1, A, K, B, N, C1, N);
      end = clock();
      total_time3 += ((double)(end - start))*1000 / CLOCKS_PER_SEC ;
    }

    printf("%d,%lf,%lf,%lf\n", SIZE, total_time1 / EPOCH, total_time2 / EPOCH,
           total_time3 / EPOCH);
    free(C1);
  }

  //   assert(memcmp(C1, C2, M * N * sizeof(float)) == 0);

  return 0;
}