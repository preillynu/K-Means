/*
   The interface is using kmeans in rodinia 3.0 benchmarks.
*/
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
//#include <math.h>
#include <iostream>

#define FLT_MAX 3.40282347e+38                                                  

#include "io.h"
#include "kmeans.h"
#include "kmeans_cpu.h"

void run_gpu(Kmeans &kmeans);

void run_gpu(Kmeans &kmeans) 
{
	int min_nclusters = kmeans.get_maxnclusters();
	int max_nclusters = kmeans.get_maxnclusters();
	int npoints       = kmeans.get_npoints();
	int nloops        = kmeans.get_nloops(); 
	int nfeatures     = kmeans.get_nfeatures();
	float threshold   = kmeans.get_threshold();

	float *data     = kmeans.get_data();
	int *membership = kmeans.get_membership();

	// use pinned memory
	//float *gpu_data;
	//cudaMallocHost(gpu_data);

	// kernel
	// note: try break the computation into different kernels to achieve better occupancy for perf

	// note: tradeoff of using constant memory and read-only l1 cache
	//			there is the overhead of copying using constant mem
	dim3 blkDim = dim3(16,16,1);

	shared_memory_size = blkDim.x * C;

	// data (N X D) ,   centers (D x C)
	__global__ void kernel_dist(float *data, float *centers, int *membership, 
			int *delta, int *membership_hist, float *centers_new_sum,
			int N, int D, int C)
	{
		extern __shared__ float sdata[];
		// rows
		int gx = threadIdx.x + blockDim.x * blockIdx.x; 
		int lx = threadIdx.x;
		// cols
		int gy = threadIdx.y + blockDim.y * blockIdx.y; 
		int ly = threadIdx.y;

		int label = -1;
	
		if(gx < N && gy < C )
		{
			float  dist = 0.f;
			// iterate through the columns	
			for(int i = 0; i<D; i++)
			{
				float tmp = data[gx * D + i] -  centers[i * D + gy];
				dist += tmp*tmp;
			}
			// save the dist matix locally
			sdata[lx * C + ly] = dist;
		}
		else {
			sdata[lx * C + ly] = FLT_MAX;
		}

		__syncthreads();

		// parallel reduction on the shared memory: on one dimension
		// notes: what if the data dims > block size
		// not opt here
		if(lx ==0 and ly==0)
		{
			float min = FLT_MAX; 
			for(int i=0; i<C; i++)
				if(min > sdata[i]) {
					min = sdata[i];
					label = i;
				}

			// compare with the previous membership
			// delta can be locally accumulated
			// update membership
			if(membership[gx] != label) {
				delta = 1;
				membership[gx] = 1.f;
			}

			// atomic add on the membership histgram
			atomAdd(&membership_hist[gx], 1);

			atomAdd(&centers_new_sum[label], data[i]);

		}
	}

	// recompute centers

}

int main(int argc, char **argv)
{
	Kmeans kmeans;

	readcmdline(argc, argv, kmeans);

	kmeans.ReadDataFromFile();

	kmeans.print_param();

	run_cpu(kmeans);

	return 0;
}
