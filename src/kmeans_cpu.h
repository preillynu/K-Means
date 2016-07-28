#ifndef _KMEANS_CPU_H_
#define _KMEANS_CPU_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

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

//----------------------------------------------------------------------------//
// CPU Kmeans Class 
//----------------------------------------------------------------------------//
class KmeansCpu {
public:
	KmeansCpu();
	~KmeansCpu();

	void print_param() const;
	void ReadDataFromFile();

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

	float 			*data;
	int 			*membership;
};

KmeansCpu::KmeansCpu() : 
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
	data = NULL;
	membership = NULL;
}

KmeansCpu::~KmeansCpu() {
	if(data != NULL)
		free(data);

	if(membership != NULL)
		free(membership);
}

void KmeansCpu::print_param() const
{
	std::cout << "threshold : "     << threshold << std::endl;
	std::cout << "max_ncluster : "  << max_nclusters << std::endl;
	std::cout << "min_nclusters : " << min_nclusters << std::endl;
	std::cout << "best_nclusters : " << best_nclusters << std::endl;

	std::cout << "nfeatures : " << nfeatures << std::endl;
	std::cout << "npoints : " << npoints << std::endl;
	std::cout << "nloops : " << nloops << std::endl;
	std::cout << "isRMSE : " << isRMSE << std::endl;
	std::cout << "rmse : " << rmse << std::endl;
	std::cout << "isOutput : " << isOutput << std::endl;

	std::cout << "filename : "      << filename << std::endl;
}

//---------------------------------------------------------------------------//
// read data from input file
//---------------------------------------------------------------------------//
void KmeansCpu::ReadDataFromFile()
{
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
	//printf("=> %d\n", npoints);

	// error check for clusters
	if (npoints < min_nclusters){
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", 
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
	//printf("=> %d\n", nfeatures);

	data = (float*) malloc(npoints * nfeatures * sizeof(float));

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

	// use -1 as default 
	membership = (int*) malloc(npoints * INT_SIZE);
	//memset(membership, -1, npoints * INT_SIZE);
	//for (int i=0; i < npoints; i++) { 
	//	std::cout << membership[i] << std::endl;
	//}

	printf("\nNumber of objects: %d\n"
			"Number of features: %d\n", npoints, nfeatures);	
}



//----------------------------------------------------------------------------//
// Functions
//----------------------------------------------------------------------------//
void run_cpu(KmeansCpu &kmeans);
void run_kmeans_cpu(int nclusters, int nfeatures, int npoints,  
		        float *data, int *membership, float *centers, float &delta); 



void run_cpu(KmeansCpu &kmeans)
{
	int min_nclusters = kmeans.min_nclusters;
	int max_nclusters = kmeans.max_nclusters;
	int npoints       = kmeans.npoints;
	int nloops        = kmeans.nloops; 
	int nfeatures     = kmeans.nfeatures;
	float threshold   = kmeans.threshold;

	float *data     = kmeans.data;
	int *membership = kmeans.membership;


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

		memset(membership, -1, npoints * INT_SIZE);

		// pick the first nclusters samples as the initial clusters 
		for(int i=0; i<nclusters; i++) {
			for(int j=0; j<nfeatures; j++) {
				centers[i * nfeatures + j] = data[i * nfeatures + j]; 
			}
		}

		// reset membership if needed
		/*
		for(int i=0; i<nclusters; i++) {
			for(int j=0; j<nfeatures; j++) {
				std::cout << centers[i * nfeatures + j] << "\t";
			}
			std::cout << std::endl;
		}
		*/

		do {
			delta = 0.f;
			run_kmeans_cpu(nclusters, nfeatures, npoints, data, membership, centers, delta);
		} while((delta>threshold) && (++loop < nloops));


		/*
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
		   */

		// free
		if(centers != NULL) free(centers);
	}
}

void run_kmeans_cpu(int nclusters, int nfeatures, int npoints,  
		        float *data, int *membership, float *centers, float &delta) 
{
	// initialize with zeros
	float* new_centers = (float*) calloc(nclusters * nfeatures, sizeof(float));
	float* new_centers_members = (float*) calloc(nclusters, sizeof(float));

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
			delta += 1.f; // atomic
			membership[i] = id;
		}

		// update membership   
		new_centers_members[id] += 1.f; // atomic
		//std::cout << id << std::endl;

		// accumulate sum for each center for each feature dim
		// hist on the center sum
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
}

#endif
