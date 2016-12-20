#ifndef _KMEANS_GPU_H_
#define _KMEANS_GPU_H_

#include <stdio.h>                                                              
#include <stdlib.h>                                                             
#include <iostream> 

#include <cuda_runtime.h>
#include <helper_cuda.h>

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


// 16K floats
__constant__ float clusters_cnst[16384];

inline int BLK(int number, int blksize)                                         
{                                                                               
	return (number + blksize - 1) / blksize;                                    
} 


__global__ void kernel_dist(const float* __restrict__ data,
		const int* __restrict__ membership,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		const int warps_per_blk,
		float* delta,
		int* new_membership,
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
	int 			*new_membership;
	float           *delta;
	float           *clusters;
	float           *new_clusters;
	float           *new_clusters_members;

	size_t membership_bytes;

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
	new_membership = NULL;
	delta      = NULL;
	clusters    = NULL;
	new_clusters    = NULL;
	new_clusters_members = NULL;

	// create timer
	checkCudaErrors( cudaEventCreate(&startEvent) );                            
	checkCudaErrors( cudaEventCreate(&stopEvent) );
}

KmeansGPU::~KmeansGPU() {
	if(data       != NULL)                 cudaFree(data);
	if(membership != NULL)                 cudaFree(membership);
	if(new_membership != NULL)             cudaFree(new_membership);
	if(delta      != NULL)                 cudaFree(delta);
	if(clusters   != NULL)                 cudaFree(clusters);
	if(new_clusters != NULL)               cudaFree(new_clusters);
	if(new_clusters_members != NULL)       cudaFree(new_clusters_members);
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

	membership_bytes = npoints * INT_SIZE;

	cudaMallocManaged((void**)&membership,     membership_bytes);	
	cudaMallocManaged((void**)&new_membership, membership_bytes);	
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

		int clusters_datapoints = nclusters * nfeatures;
		size_t clusters_bytes = clusters_datapoints * FLT_SIZE;

		//--------------------------------------------------------------------//
		// for different number of clusters, clusters need to be allocated       
		//--------------------------------------------------------------------//
		cudaMallocManaged((void**)&clusters,             clusters_bytes);
		cudaMallocManaged((void**)&new_clusters,         clusters_bytes);
		cudaMallocManaged((void**)&new_clusters_members, nclusters * FLT_SIZE);


		//--------------------------------------------------------------------//
		// pick the first [nclusters] samples as the initial clusters           
		//--------------------------------------------------------------------//
		for(int i=0; i<nclusters; i++) {                                        
			for(int j=0; j<nfeatures; j++) {                                    
				clusters[i * nfeatures + j] = data[i * nfeatures + j];           
			}                                                                   
		}                                                                       

		//----------------------//
		// copy clusters to contant memory
		//----------------------//
		cudaMemcpyToSymbol(clusters_cnst, clusters, clusters_bytes, 0, cudaMemcpyHostToDevice);

		// the membership is intialized with 0 
		cudaMemset(membership, 0, membership_bytes);

		for(int loop=0; loop<nloops; loop++)
		{
			delta[0] = 0.f;

			runKmeans_gpu(nclusters);
		
			if(delta[0] < threshold)
				break;

			// update membership
			cudaMemcpy(membership, new_membership, membership_bytes, cudaMemcpyDeviceToDevice);

			// update clusters in the constant memsory 
			cudaMemcpyToSymbol(clusters_cnst, clusters, clusters_bytes, 0, cudaMemcpyHostToDevice);

			//printf("loop: %d \t delta : %f\n", loop, delta[0]);
		}


		//--------------------------------------------------------------------//
		// release resources                                                    
		//--------------------------------------------------------------------//
		if(clusters != NULL) cudaFree(clusters);                                      
		if(new_clusters != NULL) cudaFree(new_clusters);
		if(new_clusters_members != NULL) cudaFree(new_clusters_members);
	}

	cudaEventRecord(stopEvent);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&gpuTime, startEvent, stopEvent);
	printf("GPU Elapsed Time : %f ms\n", gpuTime);
}


//----------------------------------------------------------------------------//
// Run Kmeans : GPU Kernels
//----------------------------------------------------------------------------//
void KmeansGPU::runKmeans_gpu(int nclusters)
{
	// start from zero for each iteration
	cudaMemset(new_clusters,         0, nclusters * nfeatures * FLT_SIZE);
	cudaMemset(new_clusters_members, 0, nclusters * FLT_SIZE);

	// each point working on computing the closest clusters
	int blocksize = 128;
	int warps_per_blk = blocksize >> 5;

	blkDim = dim3(blocksize, 1, 1);
	grdDim = dim3(BLK(npoints, blocksize), 1, 1);

	size_t sharedmem_size = nclusters * (nfeatures + 1) * FLT_SIZE;

	kernel_dist <<< grdDim, blkDim, sharedmem_size >>> (data, 
			membership, 
			npoints, 
			nfeatures, 
			nclusters, 
			warps_per_blk,
			delta,
			new_membership,
			new_clusters,
			new_clusters_members);

	cudaDeviceSynchronize();

	// update the clusters locally
	for(int k=0; k<nclusters; k++) {                                        
		for(int f=0; f<nfeatures; f++) {                                    
			clusters[k * nfeatures + f] = new_clusters[k * nfeatures + f] / new_clusters_members[k];           
			//printf("%f ", clusters[i * nfeatures + j]);
		}                                                                   
		//printf("\n");
	}                                                                       
}

__global__ void kernel_dist(const float* __restrict__ data,
		const int* __restrict__ membership,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		const int warps_per_blk,
		float* delta,
		int* new_membership,
		float *new_clusters,
		float *new_clusters_members)
{
	//extern __shared__ float local_cluster[]; 

	// assume 32 warps = 1024 block size
	__shared__ float warp_sm[32];
	__shared__ float feat_sm[32];
	__shared__ float change_sm[32];

	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	uint lx = threadIdx.x;

	float dist_min = FLT_MAX;
	int prev_id, curr_id;
	//size_t baseInd       = lx * (nfeatures + 1);
	size_t data_base_ind = gx * nfeatures;
	size_t center_base_ind;
	
	int my_membership = -1;
	float change = 0.f;

	if(gx < npoints) {

		// load the membership
		curr_id = prev_id = membership[gx];

		// go through each cluster
		for(int k=0; k<nclusters; k++)
		{
			float dist_cluster = 0.f;

			center_base_ind = k * nfeatures;

			// each feature dim
			for(int f=0; f<nfeatures; f++)
			{
				float diff = data[data_base_ind + f] - clusters_cnst[center_base_ind + f];
				dist_cluster += diff * diff;
			}

			// update the id for the closest center
			if(dist_cluster < dist_min) {                                       
				dist_min = dist_cluster;                                        
				curr_id = k;                                                         
			}  
		}

		//--------------------------------------------------------------------//
		// update membership
		//--------------------------------------------------------------------//
		if(prev_id != curr_id) {
			my_membership = curr_id;
			new_membership[gx] = curr_id;	// update
			change = 1.f;
		}
	}


	int lane_id = threadIdx.x & 0x1F;
	int warp_id = threadIdx.x>>5;


	//---------------------------------------------------------------//
	// update delta
	//---------------------------------------------------------------//
	#pragma unroll                                                      
	for (int i=16; i>0; i>>=1) change += __shfl_down(change, i, 32);
	if(lane_id == 0) change_sm[warp_id] = change;
	__syncthreads();
	if(warp_id == 0) {
		change = (lx < warps_per_blk) ? change_sm[lx] : 0;	
		#pragma unroll
		for (int i=16; i>0; i>>=1) change += __shfl_down(change, i, 32);
		if(lx == 0) {
			atomicAdd(&delta[0], change);	
		}
	}


	for(int k=0; k<nclusters; k++)
	{
		int   flag = 0;
		float tmp  = 0.f;			// for membership number
		if(my_membership == k) {
			flag = 1;
			tmp  = 1.f;	
		}

		//-------------------------------------------------------------//
		// counter the members for current cluster 
		//-------------------------------------------------------------//
		// warp reduction 
#pragma unroll                                                      
		for (int i=16; i>0; i>>=1 ) {                                       
			tmp += __shfl_down(tmp, i, 32);
		}

		if(lane_id == 0) {
			warp_sm[warp_id] = tmp;
		}

		__syncthreads();

		if(warp_id == 0) {
			tmp = (lx < warps_per_blk) ? warp_sm[lx] : 0.f;	

#pragma unroll
			for (int i=16; i>0; i>>=1) {                                       
				tmp += __shfl_down(tmp, i, 32);
			} 

			if(lx == 0) { 	// add the local count to the global count
				atomicAdd(&new_clusters_members[k], tmp);	
			}
		}



		//------------------------------------------------------------//
		// accumuate new clusters for each feature dim
		//------------------------------------------------------------//
		float feat;
		for(int f=0; f<nfeatures; f++) 
		{
			
			// load feature value for current data point
			feat = 0.f;
			if(flag == 1) {
				feat = data[gx * nfeatures + f];
			}

			//------------------------------------//
			// reduction for feature values
			//------------------------------------//
			// sum current warp
			#pragma unroll                                                      
			for (int i=16; i>0; i>>=1) feat += __shfl_down(feat, i, 32);
			// save warp sum to shared memory using the 1st lane
			if(lane_id == 0) feat_sm[warp_id] = feat;
			__syncthreads();
			// use the 1st warp to accumulate the block sum
			if(warp_id == 0) {
				feat = (lx < warps_per_blk) ? feat_sm[lx] : 0.f;	
				#pragma unroll
				for (int i=16; i>0; i>>=1) feat += __shfl_down(feat, i, 32);
				// add the local count to the global count
				if(lx == 0) {
					atomicAdd(&new_clusters[k * nfeatures + f], feat);	
				}
			}
		}

	}
}



#endif
