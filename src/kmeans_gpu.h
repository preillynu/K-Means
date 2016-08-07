#ifndef _KMEANS_GPU_H_
#define _KMEANS_GPU_H_

#include <stdio.h>                                                              
#include <stdlib.h>                                                             
#include <iostream> 

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_LINE_LEN 4096                                                       
                                                                                
#ifndef FLT_MAX                                                                 
#define FLT_MAX 3.40282347e+38                                                  
#endif                                                                          
                                                                                
#ifndef INT_SIZE                                                                
#define INT_SIZE sizeof(int)                                                    
#endif                                                                          
                                                                                
#ifndef FLT_SIZE                                                                
#define FLT_SIZE sizeof(float)                                                  
#endif 

static __device__ __forceinline__ int BLK(int number, int blksize)                                         
{                                                                               
	return (number + blksize - 1) / blksize;                                    
} 

// gpu kernel to initialize data
__global__ void kernel_memset (float *array, 
	const float value, 
	const int N);

__global__ void kernel_dist(const float* __restrict__ data,
		const float* __restrict__ clusters,
		int *membership,
		float *delta,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		float* new_clusters,
		float* new_clusters_members);

__global__ void kernel_update_clusters (float *clusters, 
		const float* __restrict__ new_clusters,
		const float* __restrict__ new_clusters_members,
		const int nclusters,
		const int nfeatures);

__global__ void kernel_parent_kmeans(float* data,
		float *clusters,
		int  *membership,
		float *delta,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		const float threshold,
		const int nloops,
		float *new_clusters,
		float *new_clusters_members);


//----------------------------------------------------------------------------//
// GPU Kmeans Class
//----------------------------------------------------------------------------//
class KmeansGPU {
public:
	KmeansGPU();
	~KmeansGPU();

	void print_param() const;
	void ReadDataFromFile();
	void run();
	void runKmeans_gpu(int nclusters);

	char 			*filename;
	float 			threshold;
	int 			max_nclusters;
	int 			min_nclusters;
	int 			best_nclusters;
	int 			nfeatures;
	int 			npoints;
	int 			nloops;
	int 			isRMSE;
	float 			rmse;
	int 			isOutput;
	char 			line[MAX_LINE_LEN];

	// using unified memory
	float 			*data;
	int 			*membership;
	float           *delta;
	float           *clusters;
	float           *new_clusters;
	float           *new_clusters_members;

	dim3            blkDim;
	dim3            grdDim;

	cudaEvent_t startEvent, stopEvent;
	float gpuTime;
};

KmeansGPU::KmeansGPU() : 
	filename(NULL),
	threshold(0.001f),
	max_nclusters(5),
	min_nclusters(5),
	best_nclusters(0),
	nfeatures(0),
	npoints(0),
	nloops(1000),
	isRMSE(0),
	rmse(0.f),
	isOutput(0)
{
	data       = NULL;
	membership = NULL;
	delta      = NULL;
	clusters    = NULL;
	new_clusters    = NULL;
	new_clusters_members = NULL;

	// create timer
	checkCudaErrors( cudaEventCreate(&startEvent) );                            
	checkCudaErrors( cudaEventCreate(&stopEvent) );
}

KmeansGPU::~KmeansGPU() {
	if(data       != NULL)                 cudaFreeHost(data);
	if(membership != NULL)                 cudaFreeHost(membership);
	if(delta      != NULL)                 cudaFreeHost(delta);
	if(clusters   != NULL)                 cudaFreeHost(clusters);
	if(new_clusters != NULL)               cudaFreeHost(new_clusters);
	if(new_clusters_members != NULL)       cudaFreeHost(new_clusters_members);
}

void KmeansGPU::print_param() const
{
	std::cout << "----------------------\n"                                     
		         "Kmeans Configurations:\n"                                     
		         "----------------------\n";                                    
	std::cout << "filename : "       << filename << std::endl;                  
	std::cout << "threshold : "      << threshold << std::endl;                 
	std::cout << "max_ncluster : "   << max_nclusters << std::endl;             
	std::cout << "min_nclusters : "  << min_nclusters << std::endl;             
	std::cout << "best_nclusters : " << best_nclusters << std::endl;            
	std::cout << "nfeatures : "      << nfeatures << std::endl;                 
	std::cout << "npoints : "        << npoints << std::endl;                   
	std::cout << "nloops : "         << nloops << std::endl;                    
	std::cout << "isRMSE : "         << isRMSE << std::endl;                    
	std::cout << "rmse : "           << rmse << std::endl;                      
	std::cout << "isOutput : "       << isOutput << std::endl;                  
	std::cout << "\n"; 
}

//----------------------------------------------------------------------------//
// read data from input file
//----------------------------------------------------------------------------//
void KmeansGPU::ReadDataFromFile()
{
	npoints   = 0;
	nfeatures = 0;

	FILE *infile;
	if ((infile = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "Error: no such file (%s)\n", filename);
		exit(1);
	}		
	// read data points
	while (fgets(line, MAX_LINE_LEN, infile) != NULL) {
		if (strtok(line, " \t\n") != NULL)
			npoints++;			
	}
	rewind(infile);
	// error check for clusters
	if (npoints < min_nclusters){
		fprintf(stderr, "Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", 
				min_nclusters, npoints);
		exit(1);
	}
	// read feature dim
	while (fgets(line, MAX_LINE_LEN, infile) != NULL) {
		if (strtok(line, " \t\n") != NULL) {
			nfeatures++;
			while (strtok(NULL, " ,\t\n") != NULL)
				nfeatures++;
			break;
		}
	}        
	rewind(infile);

	if(data != NULL)          cudaFreeHost(data);
	cudaMallocManaged((void **)&data, npoints * nfeatures * FLT_SIZE);

	int sample_num = 0;
	char *token;
	while (fgets(line, MAX_LINE_LEN, infile) != NULL) {
		token = strtok(line, " ,\t\n"); 
		while( token != NULL ) {
			//printf( " %s\n", token);
			data[sample_num] = atof(token);
			sample_num++;
			token = strtok(NULL, " ,\t\n");
		}
	}
	fclose(infile);
	//printf("=> %d\n", sample_num);

	cudaMallocManaged((void**)&membership, npoints * INT_SIZE);	
	cudaMallocManaged((void**)&delta, FLT_SIZE);	
}


//----------------------------------------------------------------------------//
// Run Kmeans
//----------------------------------------------------------------------------//
void KmeansGPU::run()                                                
{
	cudaEventRecord(startEvent);

	// search the best clusters for the input data                              
	for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) 
	{                                                                           
		if (nclusters > npoints) {                                              
			fprintf(stderr,                                                     
					"Can't have more clusters (%d) than the points (%d)!\n",    
					nclusters, npoints);                                        
			exit(1);                                                            
		}                                                                       
		//--------------------------------------------------------------------//
		// for different number of clusters, clusters need to be allocated       
		// the membership is intialized with -1                                 
		//--------------------------------------------------------------------//
		cudaMallocManaged((void**)&clusters,             nclusters * nfeatures * FLT_SIZE);
		cudaMallocManaged((void**)&new_clusters,         nclusters * nfeatures * FLT_SIZE);
		cudaMallocManaged((void**)&new_clusters_members, nclusters * FLT_SIZE);

		cudaMemset(membership, -1, npoints * INT_SIZE);

		//--------------------------------------------------------------------//
		// pick the first [nclusters] samples as the initial clusters           
		//--------------------------------------------------------------------//
		for(int i=0; i<nclusters; i++) {                                        
			for(int j=0; j<nfeatures; j++) {                                    
				clusters[i * nfeatures + j] = data[i * nfeatures + j];           
			}                                                                   
		}                                                                       

		// run kmeans on gpu using DP
		kernel_parent_kmeans <<< 1 , 1 >>> (data, 
				clusters, 
				membership, 
				delta,
				npoints, 
				nfeatures,
				nclusters,
				threshold,
				nloops,
				new_clusters,
				new_clusters_members
				);



	//	int loop = 0;
	//	do {
	//		delta[0] = 0.f;
	//		runKmeans_gpu(nclusters);
	//		//printf("loop: %d \t delta : %f\n", loop, delta[0]);
	//	} while((delta[0] > threshold) && (++loop < nloops));

		//--------------------------------------------------------------------//
		// release resources                                                    
		//--------------------------------------------------------------------//
		if(clusters != NULL) cudaFreeHost(clusters);                                      
		if(new_clusters != NULL) cudaFreeHost(new_clusters);
		if(new_clusters_members != NULL) cudaFreeHost(new_clusters_members);
	}

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
	printf("GPU Elapsed Time : %f ms\n", gpuTime);
}

//----------------------------------------------------------------------------//
// Run Kmeans : GPU Kernels
//----------------------------------------------------------------------------//
/*
void KmeansGPU::runKmeans_gpu(int nclusters)
{
	// start from zero for each iteration
	cudaMemset(new_clusters,         0, nclusters * nfeatures * FLT_SIZE);
	cudaMemset(new_clusters_members, 0, nclusters * FLT_SIZE);

	// each point working on computing the closest clusters
	blkDim = dim3(256, 1, 1);
	grdDim = dim3(BLK(npoints, 256), 1, 1);

	size_t sharedmem_size = nclusters * (nfeatures + 1) * FLT_SIZE;
	kernel_dist <<< grdDim, blkDim, sharedmem_size >>> (data, 
			clusters, 
			membership, 
			npoints, 
			nfeatures, 
			nclusters, 
			delta,
			new_clusters,
			new_clusters_members);

	cudaDeviceSynchronize();

	for(int i=0; i<nclusters; i++) {                                        
		for(int j=0; j<nfeatures; j++) {                                    
			clusters[i * nfeatures + j] = new_clusters[i * nfeatures + j] / new_clusters_members[i];           
			//printf("%f ", clusters[i * nfeatures + j]);
		}                                                                   
		//printf("\n");
	}                                                                       
}
*/



__global__ void kernel_parent_kmeans(float* data,
		float *clusters,
		int  *membership,
		float *delta,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		const float threshold,
		const int nloops,
		float *new_clusters,
		float *new_clusters_members)
{
	for(int loop=0; loop<nloops; loop++)
	{
		//printf("loop id : %d\n", loop);

		delta[0] = 0.f;		
		
		kernel_memset <<< 1, nclusters * nfeatures >>> (new_clusters, 0, nclusters * nfeatures);
		kernel_memset <<< 1, nclusters             >>> (new_clusters_members, 0, nclusters);

		// each point working on computing the closest clusters
		dim3 blkDim = dim3(256, 1, 1);
		dim3 grdDim = dim3(BLK(npoints, 256), 1, 1);

		size_t sharedmem_size = nclusters * (nfeatures + 1) * FLT_SIZE;
		kernel_dist <<< grdDim, blkDim, sharedmem_size >>> (data, 
				clusters, 
				membership, 
				delta,
				npoints, 
				nfeatures, 
				nclusters, 
				new_clusters,
				new_clusters_members);

/*
		for(int i=0; i<nclusters; i++) {                                        
			for(int j=0; j<nfeatures; j++) {                                    
				printf("%f ", clusters[i * nfeatures + j]);
			}                                                                   
			printf("\n");
		}                                                                       
		printf("\n\n");
		*/

		blkDim = dim3(16, 16, 1);
		grdDim = dim3(BLK(nclusters, 16), BLK(nfeatures,16), 1);
		kernel_update_clusters <<< grdDim, blkDim >>> (clusters, 
																				new_clusters, 
																				new_clusters_members,
																				nclusters,
																				nfeatures);


		// terminate condition
		if(delta[0] < threshold)
			break;
	}
}

__global__ void kernel_memset (float *array, 
	const float value, 
	const int N)
{
	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	if(gx < N)
		array[gx] = value;
}

// notes: assume nfeatures is smaller that the block size
// membership array can be avoided
__global__ void kernel_dist(const float* __restrict__ data,
		const float* __restrict__ clusters,
		int *membership,
		float *delta,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		float *new_clusters,
		float *new_clusters_members)
{
	extern __shared__ float local_cluster[]; 

	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	uint lx = threadIdx.x;

	float dist_min = FLT_MAX;
	int prev_id;
	int curr_id;
	size_t baseInd       = lx * (nfeatures + 1);
	size_t data_base_ind = gx * nfeatures;
	size_t cluster_base_ind;
	
	// initialize shared memory
	if(lx < nclusters) {
		for(int f=0; f<nfeatures+1; f++)
			local_cluster[baseInd + f] = 0.f;
	}

	__syncthreads();


	if(gx < npoints) {
		// load the membership
		curr_id = prev_id = membership[gx];

		// go through each cluster
		for(int k=0; k<nclusters; k++)
		{
			float dist_cluster = 0.f;
			size_t center_base_ind = k * nfeatures;
			for(int f=0; f<nfeatures; f++)
			{
				float diff = data[data_base_ind + f] - clusters[center_base_ind + f];
				dist_cluster += diff * diff;
			}

			// update the id for the closest center
			if(dist_cluster < dist_min) {                                       
				dist_min = dist_cluster;                                        
				curr_id = k;                                                         
			}  
		}

		// update membership
		if(prev_id != curr_id) {
			membership[gx] = curr_id; 
			atomicAdd(&delta[0], 1.f);
		}
	}

	// accumulate the data value across each dim locally 
	// accumulate the membership for each cluster locally
	if(gx < npoints)
	{
		cluster_base_ind = curr_id * (nfeatures + 1);
		for(int f=0; f<nfeatures; f++)
			atomicAdd(&local_cluster[cluster_base_ind + f], data[data_base_ind + f]);

		// member counts
		atomicAdd(&local_cluster[cluster_base_ind + nfeatures], 1.f);
	}

	__syncthreads();

	// update the global counts
	// new clusters, new cluster counts
	if(lx < nclusters)
	{
		// accumulate data globally
		size_t out_ind = lx * nfeatures;
		for(int f=0; f<nfeatures; f++)
			atomicAdd(&new_clusters[out_ind + f], local_cluster[baseInd + f]);

		// accumulate counts globally
		atomicAdd(&new_clusters_members[lx], local_cluster[baseInd + nfeatures]);
	}
}


__global__ void kernel_update_clusters (float *clusters, 
		const float* __restrict__ new_clusters,
		const float* __restrict__ new_clusters_members,
		const int nclusters,
		const int nfeatures)
{	
	uint gx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	uint gy = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

	if(gx < nclusters && gy < nfeatures)
	{
		size_t outInd = gx * nfeatures + gy;
		clusters[outInd] = new_clusters[outInd] / new_clusters_members[gx];
	}
}

#endif
