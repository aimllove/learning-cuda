// Last update: 24/12/2020
#include <stdio.h>
#include <stdint.h>

__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void printArray(uint32_t * a, int s, int e)
{
    for (int i = s; i < e; i++)
        printf("%i ", a[i]);
    printf("\n");
}

void printArrayInt(int * a, int s, int e)
{
    for (int i = s; i < e; i++)
        printf("%d ", a[i]);
    printf("\n");
}

// Sequential Radix Sort
// "const uint32_t * in" means: the memory region pointed by "in" is read-only
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
        {
            bits[i] = (src[i] >> bitIdx) & 1;
        }

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
        {
            nOnesBefore[i] = nOnesBefore[i - 1] + bits[i - 1];
        }

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__global__ void extractBitKernel(const uint32_t * src, int n,
                                int * bits,
                                int bitIdx)
{
    // PARALELLIZED BIT EXTRACTION
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
        bits[i] = int((src[i] >> bitIdx) & 1);
    __syncthreads();
}

__global__ void scanKernel(const int * bits,  int n, int * nOnesBefore,
                            volatile int * bSums)
{
    // PARALELLIZED EXCLUSIVE SCAN ON BITS
    extern __shared__ int s_data[]; // 2 * blockSize
    __shared__ int bi;
    int tx = threadIdx.x;

    if (tx == 0)
    {
        bi = atomicAdd(&bCount, 1);
    }
    __syncthreads();

    int i1, i2;
    i1 = bi * 2 * blockDim.x + tx;
    i2 = i1 + blockDim.x;

	if (i1 < n)
		s_data[threadIdx.x] = (tx == 0) ? 0 : bits[i1 - 1];
	if (i2 < n)
		s_data[threadIdx.x + blockDim.x] = bits[i2 - 1];
	__syncthreads();

    if (i1 < n)
    {
        // Each block does scan with data on SMEM
        // Reduction phase
        for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
        {
            int s_dataIdx = (tx + 1) * 2 * stride - 1; // To avoid warp divergence
            if (s_dataIdx < 2 * blockDim.x)
                s_data[s_dataIdx] += s_data[s_dataIdx - stride];
            __syncthreads();
        }
        // Post-reduction phase
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            int s_dataIdx = (tx + 1) * 2 * stride - 1 + stride; // Wow
            if (s_dataIdx < 2 * blockDim.x)
                s_data[s_dataIdx] += s_data[s_dataIdx - stride];
            __syncthreads();
        }

        // Compute bSums
        if (bSums != NULL && threadIdx.x == 0)
        {
            // bSums[bi] is the last element of s_data
            // plus
            // the bit value at corresponding position
            bSums[bi] = s_data[2 * blockDim.x - 1] + bits[(bi + 1) * 2 * blockDim.x - 1];
        }

        if (tx == 0)
        {
            if (bi > 0)
            {
                while (bCount1 < bi) {}
                bSums[bi] += bSums[bi - 1];
                __threadfence();
            }
            bCount1 += 1;
        }
        __syncthreads();

        if (bi > 0)
        {
            s_data[tx] += bSums[bi - 1];
            if(i2 < n)
                s_data[tx + blockDim.x] += bSums[bi - 1];
        }
        __syncthreads();

        nOnesBefore[i1] = s_data[tx];
        if(i2 < n)
            nOnesBefore[i2] = s_data[tx + blockDim.x];
        __syncthreads();    
    }
}

__global__ void rank_n_resultKernel(uint32_t * src, int n, uint32_t * dst,
                                    const int * bits,
                                    const int * nOnesBefore,
                                    const int nZeros)
{
    // PARALELLIZED RANK COMPUTING AND OUTPUT WRITING
    extern __shared__ int s_data[];
    int tx = threadIdx.x;
    int rank, i1, i2;

    i1 = blockIdx.x * 2 * blockDim.x + tx;
    i2 = i1 + blockDim.x;

    if(i1 < n)
    {
        s_data[tx] = nOnesBefore[i1];
        if (i2 < n)
            s_data[tx + blockDim.x] = nOnesBefore[i2];
        __syncthreads();           

        if(bits[i1] == 0)
        {
            rank = i1 - s_data[tx];
        }
        else
        {
            rank = nZeros + s_data[tx];
        }

        dst[rank] = src[i1];

        if(i2 < n)
        {
            if(bits[i2] == 0)
            {
                rank = i2 - s_data[tx + blockDim.x];
            }
            else
            {
                rank = nZeros + s_data[tx + blockDim.x];
            }

            dst[rank] = src[i2];
        }
    }
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    // Data sizes
    int blkDataSize = 2 * blockSize;
    size_t nBytes = n * sizeof(uint32_t);
    size_t helperBytes = n * sizeof(int);
    size_t smem = blkDataSize * sizeof(int);
    int gridSizeBits = (n - 1) / blockSize + 1;
    int gridSize = (n - 1) / blkDataSize + 1;

    // Default b's
    const int default_b = 0;

    uint32_t * src, * d_src, * d_dst, * temp;
    int * d_bits, * d_nOnesBefore, * d_bSums;
    int lastBit, nZeros, nOnes;
    // Host allocation
    src = (uint32_t *)malloc(nBytes);
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, nBytes);

    // Device allocation
    CHECK(cudaMalloc(&d_src, nBytes));
    CHECK(cudaMalloc(&d_dst, nBytes));
    CHECK(cudaMalloc(&temp, nBytes));
    CHECK(cudaMalloc(&d_bits, helperBytes));
    CHECK(cudaMalloc(&d_nOnesBefore, helperBytes));

    if (gridSize > 1)
    {
        CHECK(cudaMalloc(&d_bSums, gridSize * sizeof(int)));
    }
    else
    {
        d_bSums = NULL;
    }

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    CHECK(cudaMemcpy(d_src, src, nBytes, cudaMemcpyHostToDevice));
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        extractBitKernel<<<gridSizeBits, blockSize>>>(d_src, n, d_bits, bitIdx);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(&lastBit, &d_bits[n - 1], sizeof(int), cudaMemcpyDeviceToHost));

        // Compute nOnesBefore
        scanKernel<<<gridSize, blockSize, smem>>>(d_bits, n, d_nOnesBefore, d_bSums);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(&nOnes, &d_nOnesBefore[n - 1], sizeof(int), cudaMemcpyDeviceToHost));

        // Compute rank and write result
        nZeros = n - lastBit - nOnes;
        rank_n_resultKernel<<<gridSize, blockSize, smem>>>(d_src, n, d_dst, d_bits, d_nOnesBefore, nZeros);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // Reset bCount and bCount1
        CHECK(cudaMemcpyToSymbol(bCount, &default_b, sizeof(int), 0, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyToSymbol(bCount1, &default_b, sizeof(int), 0, cudaMemcpyHostToDevice));

        // Swap d_src and d_dst
        CHECK(cudaMemcpy(temp, d_src, nBytes, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(d_src, d_dst, nBytes, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(d_dst, temp, nBytes, cudaMemcpyDeviceToDevice));
    }

    // Does out array contain results?
    CHECK(cudaMemcpy(src, d_src, nBytes, cudaMemcpyDeviceToHost));
    memcpy(out, src, nBytes);

    // Free host memory
    free(originalSrc);

    // Free device memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_nOnesBefore));
    CHECK(cudaFree(d_bSums));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    // int n = 20; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        // in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, 0, 20); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);
    
    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, 0, 20); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, 0, 20); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
