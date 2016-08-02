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

inline int BLK(int number, int blksize)                                         
{                                                                               
	return (number + blksize - 1) / blksize;                                    
} 

__global__ void kernel_dist(const float* __restrict__ data,
		const float* __restrict__ centers,
		int *membership,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		float* delta);


//----------------------------------------------------------------------------//
// GPU Kmeans Class
//----------------------------------------------------------------------------//
class KmeansGPU {
public:
	KmeansGPU();
	~KmeansGPU();

	void print_param() const;
	void ReadDataFromFile();
	void runKmeans();
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
	float           *centers;
	dim3            blkDim;
	dim3            grdDim;
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
	centers    = NULL;
}

KmeansGPU::~KmeansGPU() {
	if(data != NULL)          cudaFreeHost(data);
	if(membership != NULL)    cudaFreeHost(membership);
	if(delta != NULL)         cudaFreeHost(delta);
	if(centers != NULL)       cudaFreeHost(centers);
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
	//for (int i=0; i < npoints; i++) membership[i] = -1;

	cudaMallocManaged((void**)&delta, FLT_SIZE);	
}


//----------------------------------------------------------------------------//
// Run Kmeans
//----------------------------------------------------------------------------//
void KmeansGPU::runKmeans()                                                
{
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
		// for different number of clusters, centers need to be allocated       
		// the membership is intialized with -1                                 
		//--------------------------------------------------------------------//
		cudaMallocManaged((void**)&centers, nclusters * nfeatures * FLT_SIZE);
		cudaMemset(membership, -1, npoints * INT_SIZE);

		//--------------------------------------------------------------------//
		// pick the first [nclusters] samples as the initial clusters           
		//--------------------------------------------------------------------//
		for(int i=0; i<nclusters; i++) {                                        
			for(int j=0; j<nfeatures; j++) {                                    
				centers[i * nfeatures + j] = data[i * nfeatures + j];           
			}                                                                   
		}                                                                       

		
		// dev sync
		delta[0] = 0.f;
		runKmeans_gpu(nclusters);

		/*
		float delta;                                                            
		int loop = 0;                                                           
		do {                                                                    
			delta = 0.f;                                                        
			run_kmeans_cpu(nclusters, nfeatures, npoints, data, membership, centers, delta);
			//std::cout << " loop : " << loop << std::endl;                     
		} while((delta>threshold) && (++loop < nloops));                        
		*/


		//--------------------------------------------------------------------//
		// release resources                                                    
		//--------------------------------------------------------------------//
		if(centers != NULL) cudaFreeHost(centers);                                      
	}                                                        
}

//----------------------------------------------------------------------------//
// Run Kmeans : GPU Kernels
//----------------------------------------------------------------------------//
void KmeansGPU::runKmeans_gpu(int nclusters)
{
	//------------------------------------------------------------------------//
	// new centers and membership initialize with zeros                         
	//------------------------------------------------------------------------//
	//float* new_centers = (float*) calloc(nclusters * nfeatures, sizeof(float)); 
	//float* new_centers_members = (float*) calloc(nclusters, sizeof(float));  

	blkDim = dim3(32, 8, 1);
	grdDim = dim3(BLK(npoints,32), BLK(nfeatures,8), 1);
	// store intermediate data in shared memory
	size_t sharedmem_size = blkDim.x * blkDim.y * FLT_SIZE;
	kernel_dist <<< grdDim, blkDim, sharedmem_size >>> (data, 
			centers, 
			membership, 
			npoints, nfeatures, nclusters, delta);


	blkDim = dim3(256, 1, 1);
	grdDim = dim3(BLK(npoints,256), 1, 1);

}

__global__ void kernel_compute_centers(const float* __restrict__ data,
		const float* __restrict__ membership,
		npoints, nfeatures, 
		float *centers)
{
	extern __shared__ float sdata[]; // (ncluster x nfeatures + 1)

	float *member_sum = &sdata[0];

	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	uint lx = threadIdx.x;

	if(gx < npoints)
	{
		int id = membership[gx];

		atomicAdd(&center_member_sum[id], 1.f);

		for (int k=0; k<nfeatures; k++){                                        
			new_centers[id * nfeatures + k] += data[i * nfeatures + k];         
		}    
	
	}

}



// notes: assume nfeatures is smaller that the block size
// membership array can be avoided
__global__ void kernel_dist(const float* __restrict__ data,
		const float* __restrict__ centers,
		int *membership,
		const int npoints,
		const int nfeatures,
		const int nclusters,
		float* delta)
{
	extern __shared__ float sdata[]; 

	uint gx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	uint gy = threadIdx.y + __umul24(blockDim.y, blockIdx.y);
	uint lx = threadIdx.x;
	uint ly = threadIdx.y;

	float dist_min = FLT_MAX;
	int current_id;
	
	// load the membership
	if(gx < npoints && gy == 0) {
		current_id = membership[gx];
	}

	for(int k=0; k<nclusters; k++)
	{
		// compute partial distance to shared memory
		if(gx < npoints && gy < nfeatures) {
			float diff = data[gx * nfeatures + gy] - centers[k * nfeatures + gy];
			sdata[lx * blockDim.y + ly] = diff * diff;
		} else {
			sdata[lx * blockDim.y + ly] = 0.f;	
		}

		__syncthreads();

		// activate the 1st column threads
		// notes: parallel reduction if possible
		if(gx < npoints && gy == 0)
		{
			float dist_cluster = 0.f;
			int id;
			for(int i=0; i<nfeatures; i++) {
				dist_cluster += sdata[lx * blockDim.y + i]; 	
			}

			// update the id for the closest centers                            
			if(dist_cluster < dist_min) {                                       
				dist_min = dist_cluster;                                        
				id = k;                                                         
			}  

			if(current_id != id) {
				// atomic add to the global memory	
				atomicAdd(&delta[0], 1.f);
				//membership[gx] = id;
				current_id = id;
			}
		}
	}

	// update membership at last
	if(gx < npoints && gy == 0) {
		membership[gx] = current_id;
	}
}




#endif
