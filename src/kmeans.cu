/*
   The interface is using kmeans in rodinia 3.0 benchmarks.
*/
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
//#include <math.h>

#include <iostream>

#include "io.h"
#include "kmeans.h"

#ifndef FLT_MAX                                                                 
#define FLT_MAX 3.40282347e+38                                                  
#endif  

void run_cpu(Kmeans &kmeans);
void run_kmeans_cpu(Kmeans &kmeans, int nclusters, int nfeatures, int npoints,
		float *data, int *membership, float *centers, float &delta);

//void run_gpu(Kmeans &kmeans);




int main(int argc, char **argv)
{
	Kmeans kmeans;

	readcmdline(argc, argv, kmeans);

	kmeans.ReadDataFromFile();

	kmeans.print_param();

	run_cpu(kmeans);


	return 0;
}


void run_cpu(Kmeans &kmeans)
{
	int min_nclusters = kmeans.get_maxnclusters();
	int max_nclusters = kmeans.get_maxnclusters();
	int npoints       = kmeans.get_npoints();
	int nloops        = kmeans.get_nloops(); 
	int nfeatures     = kmeans.get_nfeatures();
	float threshold   = kmeans.get_threshold();

	float *data     = kmeans.get_data();
	int *membership = kmeans.get_membership();
	//float *centers  = kmeans.get_centers();


	// search the best clusters for the input data
	for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)       
	{
		if (nclusters > npoints) {
			fprintf(stderr,
					"Can't have more clusters (%d) than the points (%d)!\n",
					nclusters, npoints);	
			exit(1);
		}

		float delta;
		int loop = 0;
		float *centers= (float*) malloc(nclusters*nfeatures*sizeof(float)); 

		// pick the first nclusters samples as the initial clusters 
		for(int i=0; i<nclusters; i++) {
			for(int j=0; j<nfeatures; j++) {
				centers[i * nfeatures + j] = data[i * nfeatures + j]; 
			}
		}

		// reset membership if needed

		for(int i=0; i<nclusters; i++) {
			for(int j=0; j<nfeatures; j++) {
				std::cout << centers[i * nfeatures + j] << "\t";
			}
			std::cout << std::endl;
		}

		do {
			delta = 0.f;
			run_kmeans_cpu(kmeans, nclusters, nfeatures, npoints, data, 
					membership, centers, delta);
		} while((delta>threshold) && (++loop < nloops));


		std::cout << "\nafter\n";
		for(int i=0; i<nclusters; i++) {
			for(int j=0; j<nfeatures; j++) {
				std::cout << centers[i * nfeatures + j] << "\t";
			}
			std::cout << std::endl;
		}

		std::cout << "\nmembership\n";
		for(int i=0; i<npoints; i++)
			std::cout << membership[i] << std::endl;


		// free
		free(centers);
	}
}

void run_kmeans_cpu(Kmeans &kmeans, int nclusters, int nfeatures, int npoints,
		float *data, int *membership, float *centers, float &delta)
{
	// initialize with zeros
	float* new_centers = (float*) calloc(nclusters * nfeatures, sizeof(float));
	float* new_centers_members = (float*) calloc(nclusters, sizeof(float));

	//int* new_membership = (int *) malloc(npoints * sizeof(int));

	// update the membership by calculating the distance
	for(int i=0; i<npoints; i++)
	{
		int id = -1;
		float dist_min = FLT_MAX; 

		// find the closest center
		for (int j = 0; j<nclusters; j++)
		{
			float dist_cluster = 0.f;
			for (int k=0; k<nfeatures; k++)
			{
				float diff = data[i * nfeatures + k] - centers[j * nfeatures + k];		
				dist_cluster += diff * diff;
			}

			//dist_cluster = sqrt(dist_cluster);
			
			if(dist_cluster < dist_min) {
				dist_min = dist_cluster;
				id = j;
			}
		}
		//std::cout << id << std::endl;

		// update changed membership 
		if(membership[i] != id) {
			delta += 1.f;
			membership[i] = id;
		}

		// update membership   
		//new_membership[i] = id;
		// count the cluster members
		new_centers_members[id] += 1.f;
		//std::cout << id << std::endl;

		// accumulate sum for each center for each feature dim
		for (int k=0; k<nfeatures; k++){
			new_centers[id * nfeatures + k] += data[i * nfeatures + k]; 
		}
	}

	// recompute the centers
	for (int j = 0; j<nclusters; j++)
	{
		for (int k=0; k<nfeatures; k++)
		{
			centers[j * nfeatures + k] = 
				new_centers[j * nfeatures + k] / new_centers_members[j];  
		}
	}

	free(new_centers);
	free(new_centers_members);
	//free(new_membership);
}
