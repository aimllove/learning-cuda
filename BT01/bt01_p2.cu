#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
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

void readPnm(char * fileName, 
		int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

__global__ void blurImgKernel(uchar3 * inPixels, int width, int height, 
		float * filter, int filterWidth, 
		uchar3 * outPixels)
{
	// TODO

	// Get data "row" and "column" indices
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	// The center of the filter
	int centerIdx = filterWidth >> 1;
	
	// Convolution
	if (ix < width && iy < height)
	{
		// Temporary summation
		float sumX = 0.0f, sumY = 0.0f, sumZ = 0.0f;

		// Instead of indexing the filter from 0
		// We start from -centerIdx to +centerIdx
		// This helps computing data indices more naturally
		for (int rFilter = -centerIdx; rFilter < centerIdx + 1; rFilter++)
		{
			for (int cFilter = -centerIdx; cFilter < centerIdx + 1; cFilter++)
			{
				// The filter element to be used
				// Plus centerIdx to do the right 0-indexing
				int iFilter = (rFilter + centerIdx) * filterWidth + (cFilter + centerIdx);
				
				// rPatched and cPatched are pseudo indices
				// Assume that we are extending the original image "centerIdx" wider on each edge
				int rPatched = iy + rFilter, cPatched = ix + cFilter;

				// rConv and cConv are the actual indices to be used for the convolution
				// Here we make use of the nested ternary operator in C/C++
				// If (Patched is inside the original image) then Conv = Patched (internal)
				// Else If (Patched < 0) then Conv = 0 (lower bound)
				// Esle Conv = height/width - 1 (upper bound)
				int rConv = (rPatched >= 0 && rPatched <= height - 1) ? rPatched : (rPatched < 0) ? 0 : (height - 1);
				int cConv = (cPatched >= 0 && cPatched <= width - 1) ? cPatched : (cPatched < 0) ? 0 : (width - 1);
				
				// The image element to be multiplied at this step
				int iConv = rConv * width + cConv;

				// A single multiplication
				sumX += inPixels[iConv].x * filter[iFilter];
				sumY += inPixels[iConv].y * filter[iFilter];
				sumZ += inPixels[iConv].z * filter[iFilter];
			}
		}

		// The actual output element
		int i = iy * width + ix;	
		outPixels[i].x = sumX;
		outPixels[i].y = sumY;
		outPixels[i].z = sumZ;
	}
}

void blurImg(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
		uchar3 * outPixels,
		bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		// TODO
		// The center of the filter
		int centerIdx = filterWidth >> 1;
	
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < height; c++)
			{
				// Convolution
				// Temporary summation
				float sumX = 0.0f, sumY = 0.0f, sumZ = 0.0f;

				// Instead of indexing the filter from 0
				// We start from -centerIdx to +centerIdx
				// This helps computing data indices more naturally
				for (int rFilter = -centerIdx; rFilter < centerIdx + 1; rFilter++)
				{
					for (int cFilter = -centerIdx; cFilter < centerIdx + 1; cFilter++)
					{
						// The filter element to be used
						// Plus centerIdx to do the right 0-indexing
						int iFilter = (rFilter + centerIdx) * filterWidth + (cFilter + centerIdx);
						
						// rPatched and cPatched are pseudo indices
						// Assume that we are extending the original image "centerIdx" wider on each edge
						int rPatched = r + rFilter, cPatched = c + cFilter;
						
						// rConv and cConv are the actual indices to be used for the convolution
						// Here we make use of the nested ternary operator in C/C++
						// If (Patched is inside the original image) then Conv = Patched (internal)
						// Else If (Patched < 0) then Conv = 0 (lower bound)
						// Esle Conv = height/width - 1 (upper bound)
						int rConv = (rPatched >= 0 && rPatched <= height - 1) ? rPatched : (rPatched < 0) ? 0 : (height - 1);
						int cConv = (cPatched >= 0 && cPatched <= width - 1) ? cPatched : (cPatched < 0) ? 0 : (width - 1);
						
						// The image element to be multiplied at this step
						int iConv = rConv * width + cConv;

						// A single multiplication
						sumX += inPixels[iConv].x * filter[iFilter];
						sumY += inPixels[iConv].y * filter[iFilter];
						sumZ += inPixels[iConv].z * filter[iFilter];
					}
				}

				// The actual output element
				int i = r * width + c;	
				outPixels[i].x = sumX;
				outPixels[i].y = sumY;
				outPixels[i].z = sumZ;
			}
		}
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO

		// Allocate device memories
		uchar3 *d_inPixels, *d_outPixels;
		float *d_filter;
		CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_outPixels, width * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float)))

		// Copy data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

		// Set grid size and call kernel (remember to check kernel error)
		dim3 gridSize((width - 1)/blockSize.x + 1, (height - 1)/blockSize.y + 1);
		blurImgKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_filter, filterWidth, d_outPixels);
		cudaError_t errorCode = cudaGetLastError();
		if (cudaSuccess != errorCode)
		{
			printf("Error: kernel did not run properly\n");
			printf("Code: %d, Reason: %s\n\n", errorCode, cudaGetErrorString(errorCode));
			return;
		}

		// Copy result from device memories
		CHECK(cudaMemcpy(outPixels, d_outPixels, width*height*sizeof(uchar3), cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
		CHECK(cudaFree(d_filter));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n", 
    		useDevice == true? "use device" : "use host", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char ** argv)
{
	if (argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Read correct output image file
	int correctWidth, correctHeight;
	uchar3 * correctOutPixels;
	readPnm(argv[3], correctWidth, correctHeight, correctOutPixels);
	if (correctWidth != width || correctHeight != height)
	{
		printf("The shape of the correct output image is invalid\n");
		return EXIT_FAILURE;
	}

	// Set up a simple filter with blurring effect 
	int filterWidth = 9;
	float * filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}

	// Blur input image using host
	uchar3 * hostOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	blurImg(inPixels, width, height, filter, filterWidth, hostOutPixels);
	
	// Compute mean absolute error between host result and correct result
	float hostErr = computeError(hostOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", hostErr);

	// Blur input image using device
	uchar3 * deviceOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}  
	blurImg(inPixels, width, height, filter, filterWidth, deviceOutPixels, true, blockSize);

	// Compute mean absolute error between device result and correct result
	float deviceErr = computeError(deviceOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", deviceErr);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(hostOutPixels);
	free(deviceOutPixels);
	free(filter);
}
