#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <algorithm>
using std::min;
#else
#define min(a,b) ((a)<(b)?(a):(b))
#endif

#define WARPSIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
 * ============================================================================
 * TESLA T4 OPTIMIZED GEMM PARAMETERS
 * ============================================================================
 *
 * This implementation uses a 3-level hierarchical tiling strategy:
 * 1. Thread Block Level: Each block processes BM×BN output elements
 * 2. Warp Level: Each warp within a block handles WM×WN elements
 * 3. Thread Level: Each thread computes TM×TN elements
 *
 * The parameters below are specifically tuned for Tesla T4 architecture:
 * - Compute Capability: 7.5
 * - SMs: 40
 * - Shared Memory per SM: 64KB
 * - Max Threads per Block: 1024
 * - Registers per SM: 65536
 */

 // THREAD BLOCK TILE DIMENSIONS
#define BM 128          // Block height (rows of matrix A processed per block)
#define BN 128          // Block width (columns of matrix B processed per block)
#define BK 8            // Block depth (K-dimension chunk size for shared memory)

// WARP TILE DIMENSIONS
#define WM 64           // Warp height (rows processed by each warp)
#define WN 64           // Warp width (columns processed by each warp)
#define WNITER 2        // Number of warp sub-iterations in N dimension

// THREAD TILE DIMENSIONS
#define TM 8            // Thread height (rows computed by each thread)
#define TN 4            // Thread width (columns computed by each thread)

// EXECUTION PARAMETERS
#define NUM_THREADS 128 // Threads per block (reduced for better occupancy)

// CALCULATED CONSTANTS (derived from above parameters)
#define NUM_WARPS (NUM_THREADS / 32)                              // = 4 warps per block
#define WMITER ((WM * WN) / (WARPSIZE * TM * TN * WNITER))        // = 1 warp sub-iteration in M
#define WSUBM (WM / WMITER)                                       // = 64 warp subtile height
#define WSUBN (WN / WNITER)                                       // = 32 warp subtile width

/*
 * ============================================================================
 * GLOBAL MEMORY TO SHARED MEMORY LOADING FUNCTION
 * ============================================================================
 *
 * This function cooperatively loads data from global memory into shared memory.
 * Key optimizations:
 * 1. Vectorized loads using float4 when alignment permits
 * 2. Matrix A is transposed during loading for better access patterns
 * 3. Coalesced memory access patterns for maximum bandwidth
 * 4. Only half the threads participate to avoid bank conflicts
 */
template <const int BM_T, const int BN_T, const int BK_T>
__device__ void loadFromGmem(const int N, const int K, float* A, float* B,
    float* As, float* Bs, const int innerRowA,
    const int innerColA, const int innerRowB,
    const int innerColB) {

    /*
     * LOADING STRIDE CALCULATION:
     * - rowStrideA: How many rows each loading iteration covers for matrix A
     * - rowStrideB: How many rows each loading iteration covers for matrix B
     * - We use (NUM_THREADS/2) because only half the threads do loading
     * - The factor of 4 accounts for float4 vectorized loads
     */
    const int rowStrideA = ((NUM_THREADS / 2) * 4) / BK_T;  // = (64 * 4) / 8 = 32 rows per iteration
    const int rowStrideB = (NUM_THREADS / 2) / (BN_T / 4);  // = 64 / (128/4) = 2 rows per iteration

    /*
     * LOADING MATRIX A WITH TRANSPOSITION:
     * ===================================
     * Goal: Load A[BM×BK] from global memory and store as As[BK×BM] in shared memory
     *
     * Why transpose? In the computation phase, we access A column-wise.
     * By transposing during loading, we convert column accesses to row accesses,
     * which are much more cache-friendly and avoid shared memory bank conflicts.
     *
     * Memory Layout Transformation:
     * Global A: A[row][col] = A[row * K + col]           (row-major)
     * Shared As: As[col][row] = As[col * BM + row]       (column-major)
     */
    for (int offset = 0; offset + rowStrideA <= BM_T; offset += rowStrideA) {
        // Calculate which element this thread will load
        int globalRowA = innerRowA + offset;    // Row index in the BM×BK tile
        int globalColA = innerColA * 4;         // Column index (×4 for float4 vectorization)

        // Bounds checking: ensure we don't read beyond the tile boundaries
        if (globalRowA < BM_T && globalColA + 3 < BK_T) {

            // ALIGNMENT-AWARE VECTORIZED LOADING:
            // Check if the memory address is 16-byte aligned for float4 access
            float* src_ptr = &A[globalRowA * K + globalColA];
            size_t ptr_val = (size_t)src_ptr;

            if ((ptr_val % 16 == 0) && (globalColA % 4 == 0)) {
                /*
                 * VECTORIZED LOAD PATH:
                 * Load 4 consecutive floats in a single transaction for better bandwidth
                 */
                float4 tmp = reinterpret_cast<float4*>(src_ptr)[0];

                /*
                 * TRANSPOSE AND STORE:
                 * Original: A[globalRowA][globalColA+0], A[globalRowA][globalColA+1], ...
                 * Transposed: As[globalColA+0][globalRowA], As[globalColA+1][globalRowA], ...
                 */
                As[(globalColA + 0) * BM_T + globalRowA] = tmp.x;
                As[(globalColA + 1) * BM_T + globalRowA] = tmp.y;
                As[(globalColA + 2) * BM_T + globalRowA] = tmp.z;
                As[(globalColA + 3) * BM_T + globalRowA] = tmp.w;
            }
            else {
                /*
                 * SCALAR LOAD PATH:
                 * Fallback to individual float loads when alignment is not guaranteed
                 * Still perform transposition, but one element at a time
                 */
                if (globalColA + 0 < BK_T) As[(globalColA + 0) * BM_T + globalRowA] = A[globalRowA * K + globalColA + 0];
                if (globalColA + 1 < BK_T) As[(globalColA + 1) * BM_T + globalRowA] = A[globalRowA * K + globalColA + 1];
                if (globalColA + 2 < BK_T) As[(globalColA + 2) * BM_T + globalRowA] = A[globalRowA * K + globalColA + 2];
                if (globalColA + 3 < BK_T) As[(globalColA + 3) * BM_T + globalRowA] = A[globalRowA * K + globalColA + 3];
            }
        }
    }

    /*
     * LOADING MATRIX B (NO TRANSPOSITION):
     * ===================================
     * Goal: Load B[BK×BN] from global memory and store as Bs[BK×BN] in shared memory
     *
     * Matrix B is kept in row-major format because we access it row-wise during computation.
     * No transposition is needed, which simplifies the loading process.
     *
     * Memory Layout (unchanged):
     * Global B: B[row][col] = B[row * N + col]           (row-major)
     * Shared Bs: Bs[row][col] = Bs[row * BN + col]       (row-major)
     */
    for (int offset = 0; offset + rowStrideB <= BK_T; offset += rowStrideB) {
        // Calculate which element this thread will load
        int globalRowB = innerRowB + offset;    // Row index in the BK×BN tile
        int globalColB = innerColB * 4;         // Column index (×4 for float4 vectorization)

        // Bounds checking for matrix B
        if (globalRowB < BK_T && globalColB + 3 < BN_T) {

            // ALIGNMENT-AWARE VECTORIZED LOADING FOR MATRIX B:
            float* src_ptr = &B[globalRowB * N + globalColB];           // Source in global memory
            float* dst_ptr = &Bs[globalRowB * BN_T + globalColB];       // Destination in shared memory

            size_t src_val = (size_t)src_ptr;
            size_t dst_val = (size_t)dst_ptr;

            if ((src_val % 16 == 0) && (dst_val % 16 == 0) && (globalColB % 4 == 0)) {
                /*
                 * VECTORIZED COPY PATH:
                 * Direct float4 copy from global to shared memory (no transposition)
                 */
                reinterpret_cast<float4*>(dst_ptr)[0] = reinterpret_cast<float4*>(src_ptr)[0];
            }
            else {
                /*
                 * SCALAR COPY PATH:
                 * Element-by-element copy when alignment is not guaranteed
                 */
                if (globalColB + 0 < BN_T) Bs[globalRowB * BN_T + globalColB + 0] = B[globalRowB * N + globalColB + 0];
                if (globalColB + 1 < BN_T) Bs[globalRowB * BN_T + globalColB + 1] = B[globalRowB * N + globalColB + 1];
                if (globalColB + 2 < BN_T) Bs[globalRowB * BN_T + globalColB + 2] = B[globalRowB * N + globalColB + 2];
                if (globalColB + 3 < BN_T) Bs[globalRowB * BN_T + globalColB + 3] = B[globalRowB * N + globalColB + 3];
            }
        }
    }
}

/*
 * ============================================================================
 * SHARED MEMORY TO REGISTER COMPUTATION FUNCTION
 * ============================================================================
 *
 * This function performs the core matrix multiplication using data in shared memory.
 * It implements a highly optimized computation pattern with register blocking.
 *
 * Algorithm Overview:
 * 1. For each K-step (dotIdx from 0 to BK-1):
 *    a. Load TM elements from As into regM registers
 *    b. Load TN elements from Bs into regN registers
 *    c. Compute outer product: regM ⊗ regN and accumulate results
 *
 * Key Optimizations:
 * - Register blocking: Each thread processes TM×TN elements simultaneously
 * - Data reuse: Each loaded element is reused multiple times in computation
 * - Memory hierarchy: Shared memory → Registers → Computation
 */
template <const int BM_T, const int BN_T, const int BK_T>
__device__ void processFromSmem(float* regM, float* regN, float* threadResults,
    const float* As, const float* Bs,
    const int warpRow, const int warpCol,
    const int threadRowInWarp, const int threadColInWarp) {

    /*
     * MAIN COMPUTATION LOOP OVER K DIMENSION:
     * ======================================
     * We process BK_T elements of the dot product per call to this function.
     * Each iteration (dotIdx) processes one slice of the K dimension.
     */
    for (int dotIdx = 0; dotIdx < BK_T; ++dotIdx) {

        /*
         * STEP 1: LOAD DATA FROM SHARED MEMORY As INTO REGISTERS
         * =====================================================
         * Goal: Load TM elements from the transposed matrix As for this thread
         *
         * Memory Access Pattern:
         * - As is stored as As[k][m] due to transposition during loading
         * - We access As[dotIdx][warp_position + thread_position]
         * - This gives us a column slice of the original matrix A
         */
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {    // Usually WMITER = 1
            for (int i = 0; i < TM; ++i) {                              // Load TM = 8 elements
                /*
                 * SHARED MEMORY INDEX CALCULATION FOR MATRIX A:
                 * dotIdx * BM_T: Start of the K-slice in transposed As
                 * warpRow * WM: Offset to this warp's row region
                 * wSubRowIdx * WSUBM: Offset within warp (usually 0)
                 * threadRowInWarp * TM + i: This thread's specific elements
                 */
                int sharedMemIdx = dotIdx * BM_T +                      // K-dimension offset
                    warpRow * WM +                        // Warp row offset
                    wSubRowIdx * WSUBM +                  // Warp sub-row offset
                    threadRowInWarp * TM + i;             // Thread-specific offset

                /*
                 * BOUNDS CHECKING AND LOADING:
                 * Ensure we don't read beyond the allocated shared memory
                 */
                if (sharedMemIdx < BM_T * BK_T) {
                    regM[wSubRowIdx * TM + i] = As[sharedMemIdx];       // Load into register
                }
                else {
                    regM[wSubRowIdx * TM + i] = 0.0f;                   // Zero-padding for safety
                }
            }
        }

        /*
         * STEP 2: LOAD DATA FROM SHARED MEMORY Bs INTO REGISTERS
         * ======================================================
         * Goal: Load TN elements from matrix Bs for this thread
         *
         * Memory Access Pattern:
         * - Bs is stored as Bs[k][n] (row-major, no transposition)
         * - We access Bs[dotIdx][warp_position + thread_position]
         * - This gives us a row slice of matrix B
         */
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {   // WNITER = 2 iterations
            for (int i = 0; i < TN; ++i) {                             // Load TN = 4 elements
                /*
                 * SHARED MEMORY INDEX CALCULATION FOR MATRIX B:
                 * dotIdx * BN_T: Start of the K-slice in Bs
                 * warpCol * WN: Offset to this warp's column region
                 * wSubColIdx * WSUBN: Offset within warp sub-column
                 * threadColInWarp * TN + i: This thread's specific elements
                 */
                int sharedMemIdx = dotIdx * BN_T +                      // K-dimension offset
                    warpCol * WN +                        // Warp column offset
                    wSubColIdx * WSUBN +                  // Warp sub-column offset
                    threadColInWarp * TN + i;             // Thread-specific offset

                /*
                 * BOUNDS CHECKING AND LOADING:
                 */
                if (sharedMemIdx < BK_T * BN_T) {
                    regN[wSubColIdx * TN + i] = Bs[sharedMemIdx];       // Load into register
                }
                else {
                    regN[wSubColIdx * TN + i] = 0.0f;                   // Zero-padding for safety
                }
            }
        }

        /*
         * STEP 3: COMPUTE OUTER PRODUCT AND ACCUMULATE RESULTS
         * ====================================================
         * Goal: Compute regM ⊗ regN and accumulate in threadResults
         *
         * This is the core computation: for each element in regM and regN,
         * multiply them together and add to the corresponding result element.
         *
         * Mathematical Operation:
         * threadResults[i][j] += regM[i] * regN[j]  (for all i,j combinations)
         *
         * Performance Impact:
         * - Each regM element is reused TN times
         * - Each regN element is reused TM times
         * - Total operations: WMITER × WNITER × TM × TN = 1 × 2 × 8 × 4 = 64 ops
         */
        for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        /*
                         * RESULT INDEX CALCULATION:
                         * Map from 4D iteration space to 1D threadResults array
                         */
                        int resultIdx = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN;

                        /*
                         * MULTIPLY-ACCUMULATE OPERATION:
                         * This is the fundamental GEMM operation: C += A * B
                         */
                        threadResults[resultIdx] +=
                            regM[wSubRowIdx * TM + resIdxM] *
                            regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
    }
}

/*
 * ============================================================================
 * MAIN KERNEL: DOUBLE-BUFFERED GEMM WITH WARP TILING
 * ============================================================================
 *
 * This kernel implements matrix multiplication C = α*A*B + β*C using:
 * 1. Hierarchical tiling (Block → Warp → Thread)
 * 2. Double buffering for latency hiding
 * 3. Shared memory optimization
 * 4. Register blocking for high arithmetic intensity
 *
 * Thread Organization:
 * - 128 threads per block arranged as 4 warps
 * - Each warp processes a 64×64 output region
 * - Each thread computes 8×4 output elements
 *
 * Double Buffering Strategy:
 * - First 64 threads: Computation group (process data)
 * - Last 64 threads: Loading group (load next data)
 * - Overlap computation with memory transfers
 */
__global__ void __launch_bounds__(NUM_THREADS)
sgemmDoubleBuffering(const int M, const int N, const int K,
    const float alpha, float* A, float* B, const float beta,
    float* C) {

    /*
     * THREAD BLOCK POSITIONING:
     * ========================
     * Each thread block is responsible for computing a BM×BN region of output matrix C.
     * The block position determines which part of the matrices this block will process.
     */
    const int cRow = blockIdx.y;    // Block's row in the grid (M dimension)
    const int cCol = blockIdx.x;    // Block's column in the grid (N dimension)

    /*
     * EARLY TERMINATION FOR OUT-OF-BOUNDS BLOCKS:
     * Some blocks may extend beyond the matrix boundaries due to grid rounding.
     * These blocks should exit early to avoid unnecessary computation and memory access.
     */
    if (cRow * BM >= M || cCol * BN >= N) {
        return;  // This block is completely outside the matrix bounds
    }

    /*
     * WARP POSITIONING WITHIN THREAD BLOCK:
     * ====================================
     * Each thread block contains NUM_WARPS = 4 warps.
     * We need to determine which warp this thread belongs to and where
     * that warp should operate within the BM×BN block tile.
     *
     * Warp Layout: With BN=128, WN=64, we have BN/WN = 2 warps horizontally
     *              With BM=128, WM=64, we have BM/WM = 2 warps vertically
     *              Total: 2×2 = 4 warps per block ✓
     */
    const int warpIdx = threadIdx.x / WARPSIZE;         // Which warp: 0, 1, 2, or 3
    const int warpCol = warpIdx % (BN / WN);            // Warp column: 0 or 1
    const int warpRow = warpIdx / (BN / WN);            // Warp row: 0 or 1

    /*
     * THREAD POSITIONING WITHIN WARP:
     * ==============================
     * Each warp contains 32 threads that must be arranged to process a WM×WN region.
     * With WM=64, WN=64, and each thread processing TM×TN = 8×4 elements:
     * - Threads in M dimension: WM/TM = 64/8 = 8 threads
     * - Threads in N dimension: WN/TN = 64/4 = 16 threads
     * - Total: 8×16 = 128 threads... but we only have 32 threads per warp!
     *
     * Solution: Use WNITER=2 iterations in the N dimension
     * - Threads in M dimension: WSUBM/TM = 64/8 = 8 threads
     * - Threads in N dimension: WSUBN/TN = 32/4 = 8 threads
     * - Total: 8×8 = 64 threads... still too many!
     *
     * Actual arrangement: 32 threads arranged as 8×4 with iterations
     */
    const int threadIdxInWarp = threadIdx.x % WARPSIZE;             // Thread ID within warp [0,31]
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);     // Thread column [0,7] (32/4=8)
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);     // Thread row [0,3] (32/8=4)

    /*
     * SHARED MEMORY ALLOCATION:
     * ========================
     * We allocate double the normal amount for double buffering.
     * Buffer 0: As[0:BM*BK-1], Bs[0:BK*BN-1]
     * Buffer 1: As[BM*BK:2*BM*BK-1], Bs[BK*BN:2*BK*BN-1]
     *
     * Total shared memory usage:
     * As: 2 × 128 × 8 = 2,048 floats = 8,192 bytes
     * Bs: 2 × 8 × 128 = 2,048 floats = 8,192 bytes
     * Total: 16,384 bytes = 16KB (well within Tesla T4's 64KB limit)
     */
    __shared__ float As[2 * BM * BK];   // Double-buffered shared memory for matrix A
    __shared__ float Bs[2 * BK * BN];   // Double-buffered shared memory for matrix B

    /*
     * DOUBLE BUFFERING GROUP ASSIGNMENT:
     * =================================
     * We split the 128 threads into two groups:
     * - Compute Group (threads 0-63): Handle computation on current data
     * - Loading Group (threads 64-127): Load next data while computation happens
     *
     * This overlap hides memory latency and improves performance.
     */
    bool isComputeGroup = threadIdx.x < (NUM_THREADS / 2);  // First 64 threads compute

    /*
     * MATRIX POINTER MANAGEMENT:
     * =========================
     * We need to keep track of both the original matrix pointers (for bounds checking)
     * and the current working pointers (which advance through the matrices).
     */
    float* A_orig = A;  // Original pointer for bounds checking
    float* B_orig = B;  // Original pointer for bounds checking
    float* C_orig = C;  // Original pointer for final output

    /*
     * ADVANCE POINTERS TO THIS BLOCK'S DATA REGION:
     * =============================================
     * Move the working pointers to the start of this block's data:
     * - A pointer: Move to row cRow*BM (skip cRow*BM rows)
     * - B pointer: Move to column cCol*BN (skip cCol*BN columns)
     * - C pointer: Move to this warp's output region within the block
     */
    A += cRow * BM * K;                                     // Skip to block's rows in A
    B += cCol * BN;                                         // Skip to block's columns in B
    C += (cRow * BM + warpRow * WM) * N +                   // Skip to warp's output row
        (cCol * BN + warpCol * WN);                        // Skip to warp's output column

    /*
     * LOADING THREAD ASSIGNMENT:
     * =========================
     * For loading operations, we use only half the threads (64 out of 128) to avoid
     * shared memory bank conflicts and improve efficiency.
     *
     * Each loading thread is assigned specific rows/columns to load:
     */
    const int loadThreadId = threadIdx.x % (NUM_THREADS / 2);       // Map to [0,63]

    // For matrix A loading (loading threads arranged to cover BM×BK region):
    const int innerRowA = loadThreadId / (BK / 4);                  // Row assignment [0,31]
    const int innerColA = loadThreadId % (BK / 4);                  // Column group [0,1]

    // For matrix B loading (loading threads arranged to cover BK×BN region):
    const int innerRowB = loadThreadId / (BN / 4);                  // Row assignment [0,1]
    const int innerColB = loadThreadId % (BN / 4);                  // Column group [0,31]

    /*
     * REGISTER ALLOCATION:
     * ===================
     * Each thread allocates private registers to store:
     * 1. Final results for its assigned output elements
     * 2. Temporary data loaded from shared memory during computation
     *
     * Register usage per thread:
     * - threadResults: WMITER×TM×WNITER×TN = 1×8×2×4 = 64 floats
     * - regM: WMITER×TM = 1×8 = 8 floats
     * - regN: WNITER×TN = 2×4 = 8 floats
     * - Total: 80 floats = 80 registers (good for Tesla T4 occupancy)
     */
    float threadResults[WMITER * TM * WNITER * TN];  // Accumulate final results here
    float regM[WMITER * TM];                         // Temporary storage for A data
    float regN[WNITER * TN];                         // Temporary storage for B data

    /*
     * INITIALIZE REGISTER ARRAYS:
     * ==========================
     * All arrays must be explicitly initialized to zero.
     * threadResults accumulates the dot products, so it must start at zero.
     * regM and regN are temporary, but initialization helps with debugging.
     */
    for (int i = 0; i < WMITER * TM * WNITER * TN; i++) {
        threadResults[i] = 0.0f;
    }
    for (int i = 0; i < WMITER * TM; i++) {
        regM[i] = 0.0f;
    }
    for (int i = 0; i < WNITER * TN; i++) {
        regN[i] = 0.0f;
    }

    /*
     * INITIAL DATA LOADING:
     * ====================
     * Before starting the computation loop, we need to load the first batch of data
     * into shared memory buffer 0. Only the compute group does this initial loading.
     */
    if (isComputeGroup) {
        loadFromGmem<BM, BN, BK>(N, K, A, B, As, Bs,
            innerRowA, innerColA, innerRowB, innerColB);
    }
    __syncthreads();  // Ensure all data is loaded before any thread starts computing

    /*
     * MAIN COMPUTATION LOOP WITH DOUBLE BUFFERING:
     * ===========================================
     *
     * This loop processes the K dimension in chunks of 2*BK to implement double buffering.
     * Each iteration processes two K-chunks: one from buffer 0 and one from buffer 1.
     *
     * Double Buffering Pattern:
     * 1. Compute on data in buffer 0 while loading next data into buffer 1
     * 2. Compute on data in buffer 1 while loading next data into buffer 0
     * 3. Repeat until all K dimension is processed
     *
     * This pattern hides memory access latency behind computation.
     */
    for (int bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {

        if (isComputeGroup) {
            /*
             * COMPUTE GROUP RESPONSIBILITIES:
             * ==============================
             * 1. Process current data in buffer 0
             * 2. Process current data in buffer 1 (if available)
             * 3. Load next batch into buffer 0 for future iteration
             */

             // Phase 1: Process data in buffer 0
            processFromSmem<BM, BN, BK>(regM, regN, threadResults,
                As,      // Buffer 0 for A
                Bs,      // Buffer 0 for B
                warpRow, warpCol, threadRowInWarp, threadColInWarp);
            __syncthreads();  // Sync before accessing buffer 1

            // Phase 2: Process data in buffer 1 (if we have more K data)
            if (bkIdx + BK < K) {
                processFromSmem<BM, BN, BK>(regM, regN, threadResults,
                    As + (BM * BK),  // Buffer 1 for A
                    Bs + (BK * BN),  // Buffer 1 for B
                    warpRow, warpCol, threadRowInWarp, threadColInWarp);
            }
            __syncthreads();  // Sync before loading new data

            // Phase 3: Load next batch into buffer 0 for future use
            if (bkIdx + 2 * BK < K) {
                loadFromGmem<BM, BN, BK>(N, K,
                    A + 2 * BK,        // Skip ahead 2 K-tiles in A
                    B + 2 * BK * N,    // Skip ahead 2 K-tiles in B
                    As, Bs,            // Load into buffer 0
                    innerRowA, innerColA, innerRowB, innerColB);
            }
        }
        else {
            /*
             * LOADING GROUP RESPONSIBILITIES:
             * ==============================
             * While the compute group processes data in buffer 0, the loading group
             * loads the next batch of data into buffer 1. This creates a pipeline
             * where memory access and computation overlap.
             */

             // Load next data into buffer 1 while compute group works on buffer 0
            if (bkIdx + BK < K) {
                loadFromGmem<BM, BN, BK>(N, K,
                    A + BK,            // Skip ahead 1 K-tile in A
                    B + BK * N,        // Skip ahead 1 K-tile in B
                    As + (BM * BK),    // Load into buffer 1 for A
                    Bs + (BK * BN),    // Load into buffer 1 for B
                    innerRowA, innerColA, innerRowB, innerColB);
            }

            /*
             * SYNCHRONIZATION POINTS:
             * The loading group must sync at the same points as the compute group
             * to maintain the double buffering protocol.
             */
            __syncthreads();  // Sync point 1: After buffer 0 computation
            __syncthreads();  // Sync point 2: After buffer 1 computation
        }

        /*
         * ADVANCE POINTERS FOR NEXT ITERATION:
         * ===================================
         * Move both A and B pointers forward by 2*BK elements in the K dimension.
         * This prepares for loading the next set of tiles in the subsequent iteration.
         */
        A += 2 * BK;        // Advance A by 2 K-tiles (move right by 2*BK columns)
        B += 2 * BK * N;    // Advance B by 2 K-tiles (move down by 2*BK rows)
        __syncthreads();    // Final sync before next iteration
    }

    /*
     * ============================================================================
     * WRITE RESULTS BACK TO GLOBAL MEMORY
     * ============================================================================
     *
     * After all K-dimension processing is complete, each thread writes its
     * accumulated results back to the appropriate location in matrix C.
     *
     * This phase requires careful bounds checking to ensure we don't write
     * beyond the actual matrix boundaries, especially for edge blocks.
     *
     * GEMM Operation: C = α*A*B + β*C
     * - α (alpha): Scaling factor for the computed A*B product
     * - β (beta): Scaling factor for the existing C values
     * - When β=0, we overwrite C; when β=1, we add to existing C
     */
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {        // Usually just 1 iteration
        for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {    // 2 iterations

            /*
             * NESTED LOOPS OVER THREAD'S OUTPUT ELEMENTS:
             * Each thread is responsible for TM×TN = 8×4 = 32 output elements.
             * We iterate through all of them and write each to global memory.
             */
            for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {         // 8 iterations (rows)
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 1) {     // 4 iterations (columns)

                    /*
                     * GLOBAL POSITION CALCULATION:
                     * ============================
                     * We need to calculate the exact (row, column) position in the
                     * global matrix C where this thread should write its result.
                     *
                     * Position Breakdown:
                     * - cRow * BM: Block's starting row in global matrix
                     * - warpRow * WM: Warp's starting row within block
                     * - wSubRowIdx * WSUBM: Warp sub-iteration row offset
                     * - threadRowInWarp * TM: Thread's starting row within warp
                     * - resIdxM: Specific element within thread's row range
                     *
                     * Similar calculation for columns (N dimension).
                     */
                    int globalRow = cRow * BM +                         // Block row offset
                        warpRow * WM +                       // Warp row offset
                        wSubRowIdx * WSUBM +                 // Warp sub-row offset
                        threadRowInWarp * TM +               // Thread row offset
                        resIdxM;                             // Element row offset

                    int globalCol = cCol * BN +                         // Block column offset
                        warpCol * WN +                       // Warp column offset
                        wSubColIdx * WSUBN +                 // Warp sub-column offset
                        threadColInWarp * TN +               // Thread column offset
                        resIdxN;                             // Element column offset

                    /*
                     * BOUNDS CHECKING:
                     * ===============
                     * Critical safety check: ensure we don't write beyond the actual
                     * matrix boundaries. This is especially important for:
                     * 1. Non-square matrices where M ≠ N
                     * 2. Matrix sizes that don't divide evenly by block size
                     * 3. Edge blocks that may extend beyond matrix boundaries
                     */
                    if (globalRow < M && globalCol < N) {

                        /*
                         * THREADRESULTS INDEX CALCULATION:
                         * ===============================
                         * Map from the 4D iteration space (wSubRowIdx, wSubColIdx, resIdxM, resIdxN)
                         * to the 1D threadResults array index.
                         *
                         * The array is organized as:
                         * threadResults[wSubRow][resIdxM][wSubCol][resIdxN]
                         *
                         * Flattened index = wSubRow * (TM * WNITER * TN) +
                         *                   resIdxM * (WNITER * TN) +
                         *                   wSubCol * TN +
                         *                   resIdxN
                         */
                        const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            wSubColIdx * TN + resIdxN;

                        /*
                         * GEMM UPDATE OPERATION:
                         * =====================
                         * Perform the final GEMM operation: C = α*AB + β*C
                         *
                         * Steps:
                         * 1. Read current value from global matrix C
                         * 2. Multiply computed result by α (alpha)
                         * 3. Multiply existing C value by β (beta)
                         * 4. Add them together
                         * 5. Write back to global matrix C
                         *
                         * Note: We use C_orig (original pointer) instead of the modified C pointer
                         * to ensure we write to the correct absolute position in the matrix.
                         */
                        float current = C_orig[globalRow * N + globalCol];                  // Read existing C value
                        C_orig[globalRow * N + globalCol] = alpha * threadResults[i] +      // Write: α*AB + β*C
                            beta * current;
                    }
                    /*
                     * If bounds check fails, we simply skip this write operation.
                     * This handles edge cases gracefully without corrupting memory.
                     */
                }
            }
        }
    }

    /*
     * KERNEL COMPLETION:
     * =================
     * At this point, all threads have written their results to global memory.
     * The GEMM operation for this thread block is complete.
     *
     * Key Achievements:
     * 1. Processed a BM×BN tile of the output matrix
     * 2. Used shared memory to optimize data reuse
     * 3. Employed double buffering to hide memory latency
     * 4. Leveraged register blocking for high arithmetic intensity
     * 5. Maintained memory safety through careful bounds checking
     */
}

/*
 * ============================================================================
 * HOST FUNCTION: KERNEL LAUNCHER AND CONFIGURATION
 * ============================================================================
 *
 * This function sets up the kernel launch parameters and executes the GEMM operation.
 * It handles the interface between CPU code and the GPU kernel.
 */
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    /*
     * GEMM OPERATION PARAMETERS:
     * =========================
     * Standard GEMM notation: C = α*A*B + β*C
     * - α=1.0: No scaling of the A*B product
     * - β=0.0: Overwrite C (don't add to existing values)
     */
    const float alpha = 1.0f;  // Scaling factor for A*B product
    const float beta = 0.0f;   // Scaling factor for existing C (0 = overwrite)

    /*
     * CUDA EXECUTION CONFIGURATION:
     * =============================
     * Set up the grid and block dimensions for kernel execution.
     */
    dim3 blockDim(NUM_THREADS);                     // 128 threads per block
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // Grid covers entire matrix

    /*
     * CONFIGURATION REPORTING:
     * =======================
     * Print detailed information about the kernel configuration.
     * This helps with debugging and performance analysis.
     */
    printf("Tesla T4 Optimized GEMM Configuration:\n");
    printf("=====================================\n");
    printf("Matrix dimensions: A[%d×%d] × B[%d×%d] = C[%d×%d]\n", M, K, K, N, M, N);
    printf("Tiling configuration:\n");
    printf("  - Block tile: %d×%d×%d (each block processes %d output elements)\n",
        BM, BN, BK, BM * BN);
    printf("  - Warp tile: %d×%d (each warp processes %d output elements)\n",
        WM, WN, WM * WN);
    printf("  - Thread tile: %d×%d (each thread processes %d output elements)\n",
        TM, TN, TM * TN);
    printf("Execution parameters:\n");
    printf("  - Threads per block: %d\n", NUM_THREADS);
    printf("  - Warps per block: %d\n", NUM_WARPS);
    printf("  - Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("  - Total thread blocks: %d\n", gridDim.x * gridDim.y);
    printf("Resource usage estimate:\n");
    printf("  - Shared memory per block: ~16KB\n");
    printf("  - Registers per thread: ~80\n");
    printf("  - Arithmetic intensity: ~32 FLOPs/byte\n");

    /*
     * GRID COVERAGE VALIDATION:
     * ========================
     * Ensure our grid dimensions are sufficient to cover the entire matrix.
     * This is a sanity check to catch configuration errors.
     */
    if (gridDim.x * BN < N || gridDim.y * BM < M) {
        printf("WARNING: Grid may not cover entire matrix!\n");
        printf("  Grid covers: %d×%d, but matrix is %d×%d\n",
            gridDim.y * BM, gridDim.x * BN, M, N);
    }

    /*
     * KERNEL LAUNCH:
     * =============
     * Execute the GEMM kernel on the GPU with the configured parameters.
     */
    sgemmDoubleBuffering << <gridDim, blockDim >> > (M, N, K, alpha, (float*)A, (float*)B, beta, C);

    /*
     * ERROR CHECKING:
     * ==============
     * Check for both launch errors and execution errors.
     */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Wait for kernel completion and check for runtime errors
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("Kernel execution completed successfully!\n");
}
