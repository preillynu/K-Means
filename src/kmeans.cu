/*
   The interface is using kmeans in rodinia 3.0 benchmarks.

*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
//#include <math.h>

#include <iostream>

#define MAX_LINE_LEN 4096

#ifndef FLT_MAX                                                                 
  #define FLT_MAX 3.40282347e+38                                                  
#endif

void print_usage(char *argv0) {
	const char *info=
		"\nUsage: %s [switches] -i filename\n\n"
		"    -i filename      :file containing data to be clustered\n"		
		"    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
		"    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
		"    -t threshold     :threshold value                       [default=0.001]\n"
		"    -l nloops        :iteration for each number of clusters [default=1]\n"
		"    -r               :calculate RMSE                        [default=off]\n"
		"    -o               :output cluster center coordinates     [default=off]\n"
		"    -h               :print usage message\n";
	fprintf(stderr, info, argv0);
}

class Kmeans {
public:
	Kmeans();
	~Kmeans();

	void set_filename(char *);
	char *get_filename() const;

	void set_threshold(float);
	float get_threshold() const;

	void set_maxnclusters(int);
	int get_maxnclusters() const;

	void set_minnclusters(int);
	int get_minnclusters() const;

	void set_isRMSE(int);
	void set_isOutput(int);
	void set_nloops(int);

	void print() const;

	void readDataFromFile();

	int get_npoints() const;
	int get_nloops() const;
	int get_nfeatures() const;
	float *get_features() const;
	int *get_membership() const;

private:
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
	// data
	float 			*buf;
	//float 			**features;
	float 			*features;
	float 			**cluster_centers;
	int 			*membership;
	float 			*clusters;
};

Kmeans::Kmeans() : 
	filename(NULL),
	threshold(0.001f),
	max_nclusters(5),
	min_nclusters(5),
	best_nclusters(0),
	nfeatures(0),
	npoints(0),
	nloops(1),
	isRMSE(0),
	rmse(0.f),
	isOutput(0)
{
	buf = NULL;
	features = NULL;
	cluster_centers = NULL;
}

Kmeans::~Kmeans() {
	//free(features[0]);
	free(features);
	free(membership);
}

void Kmeans::set_filename(char *fname) 
{
	filename = fname;
}

char *Kmeans::get_filename() const 
{
	return filename;
}

void Kmeans::set_threshold(float thld) 
{
	threshold = thld;
}

void Kmeans::set_maxnclusters(int maxcluster)
{
	max_nclusters = maxcluster;
}

int Kmeans::get_maxnclusters() const
{
	return max_nclusters;
}

void Kmeans::set_minnclusters(int mincluster)
{
	min_nclusters = mincluster;
}

int Kmeans::get_minnclusters() const
{
	return min_nclusters;
}

void Kmeans::set_isRMSE(int rmse)
{
	isRMSE = rmse;
}

void Kmeans::set_isOutput(int output)
{
	isOutput = output;
}

void Kmeans::set_nloops(int loops)
{
	nloops = loops;
}

int Kmeans::get_npoints() const
{
	return npoints;
}

int Kmeans::get_nloops() const
{
	return nloops;
}

int Kmeans::get_nfeatures() const
{
	return nfeatures; 
}

float* Kmeans::get_features() const
{
	return features; 
}

int* Kmeans::get_membership() const
{
	return membership; 
}

float Kmeans::get_threshold() const
{
	return threshold; 
}

void Kmeans::print() const
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

void Kmeans::readDataFromFile()
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
	// error check for clusters
	if (npoints < min_nclusters){
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", 
				min_nclusters, npoints);
		exit(1);
	}
	// read feature dim
	while (fgets(line, MAX_LINE_LEN, infile) != NULL) {
		if (strtok(line, " \t\n") != NULL) {
			// ignore the id (first attribute), start from 2nd pos
			while (strtok(NULL, " ,\t\n") != NULL)
				nfeatures++;
			break;
		}
	}        
	rewind(infile);

	features   = (float*) malloc(npoints*nfeatures*sizeof(float));
	//features    = (float**)malloc(npoints*          sizeof(float*));
	//features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
	//for (int i=1; i<npoints; i++)
	//	features[i] = features[i-1] + nfeatures;

	int sample_num = 0;
	while (fgets(line, MAX_LINE_LEN, infile) != NULL) {
		if (strtok(line, " \t\n") == NULL) continue;            
		for (int j=0; j<nfeatures; j++) {
			features[sample_num] = atof(strtok(NULL, " ,\t\n"));             
			sample_num++;
		}            
	}
	fclose(infile);

	printf("\nReading file completed\n"
			"\nNumber of objects: %d\n"
			"Number of features: %d\n", npoints,nfeatures);	

	//memcpy(features[0], buf, npoints*nfeatures*sizeof(float));

	// release buf
	//free(buf);


	membership = (int*) malloc(npoints*sizeof(int));
	for (int i=0; i < npoints; i++) membership[i] = -1;
	
}

void readcmdline(int argc, char **argv, Kmeans &kmeans)
{
	int 			opt;
	extern char 	*optarg;

	if(argc == 1) {
		print_usage(argv[0]);
		exit(1);
	}

	while ( (opt=getopt(argc,argv,"i:t:m:n:l:roh"))!= EOF) {
		switch (opt) {
			case 'i':
				kmeans.set_filename(optarg);
				break;
			case 't': 
				kmeans.set_threshold(atof(optarg));
				break;
			case 'm': 
				kmeans.set_maxnclusters(atoi(optarg));
				break;
			case 'n': 
				kmeans.set_minnclusters(atoi(optarg));
				break;
			case 'r':
				kmeans.set_isRMSE(1);
				break;
			case 'o': 
				kmeans.set_isOutput(1);
				break;
			case 'l': 
				kmeans.set_nloops(atoi(optarg));
				break;
			case 'h': 
				print_usage(argv[0]);
				exit(1);
			case '?': 
				print_usage(argv[0]);
				exit(1);
		}
	}


	if(kmeans.get_filename() == NULL) {
		print_usage(argv[0]);
		exit(1);
	}

}

void run_kmeans_cpu(Kmeans &kmeans, int nclusters, int nfeatures, int npoints,
		float *features, int *membership)
{
	float deltas = 0.f;

	// init the cluster centroids
	// each feature dim has its own centroid
	float *centers= (float*) malloc(nclusters*nfeatures*sizeof(float)); 

	// pick the first nclusters samples as the initial clusters 
	for(int i=0; i<nclusters; i++) {
		for(int j=0; j<nfeatures; j++) {
			centers[i * nfeatures + j] = features[i*nfeatures + j]; 
		}
	}

	float* new_centers = (float*) calloc(nclusters * nfeatures, sizeof(float));
	float* new_centers_members = (float*) calloc(nclusters, sizeof(float));

	int* new_membership = (int *) malloc(npoints * sizeof(int));

	// update the membership by calculating the distance
	for(int i=0; i<npoints; i++)
	{
		int id = -1;
		float dist_min = FLT_MAX; 

		for (int j = 0; j<nclusters; j++)
		{
			float dist_cluster = 0.f;
			// euclidean distance 
			for (int k=0; k<nfeatures; k++)
			{
				float diff = features[i * nfeatures + k] - centers[j * nfeatures + k];		
				dist_cluster += diff * diff;
			}
			dist_cluster = sqrt(dist_cluster);
			
			if(dist_cluster < dist_min) {
				dist_min = dist_cluster;
				id = j;
			}
		}

		// update  
		if(membership[i] != id)
			deltas += 1.f;

		new_membership[i] = id;
		new_centers_members[id] += 1.f;
		//std::cout << id << std::endl;
	}

	// compute the new centers
	for(int i=0; i<npoints; i++)
	{
		int centerID = new_membership[i];

		for (int k=0; k<nfeatures; k++)
		{
			new_centers[centerID * nfeatures + k] += 
				features[i * nfeatures + k]; 
		}
	}

	for (int j = 0; j<nclusters; j++)
	{
		for (int k=0; k<nfeatures; k++)
		{
			centers[j * nfeatures + k] = 
				new_centers[j * nfeatures + k] / new_centers_members[j];  
		}
	}

	free(centers);
	free(new_centers);
	free(new_centers_members);
	free(new_membership);
}

void run_cpu(Kmeans &kmeans)
{
	int min_nclusters = kmeans.get_maxnclusters();
	int max_nclusters = kmeans.get_maxnclusters();
	int npoints = kmeans.get_npoints();
	int nloops  = kmeans.get_nloops(); 
	int nfeatures = kmeans.get_nfeatures();
	float *features = kmeans.get_features();
	int *membership = kmeans.get_membership();
	float threshold = kmeans.get_threshold();

	// search the best clusters for the input data
	for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)       
	{
		if (nclusters > npoints) {
			fprintf(stderr,
					"Can't have more clusters (%d) than the points (%d)!\n",
					nclusters, npoints);	
			exit(1);
		}

		// check the rmse error 

		float delta;
		int loop;
		for (int i = 0; i < nloops; i++)
		{
			delta = 0.f;
			loop = 0;
			do {
				run_kmeans_cpu(kmeans, 
						nclusters, 
						nfeatures, 
						npoints, 
						features, 
						membership,
						delta);
			} while((delta>threshold) && (loop++ < 500));
		
		}

	}


}


int main(int argc, char **argv)
{
	Kmeans kmeans;

	readcmdline(argc, argv, kmeans);

	kmeans.readDataFromFile();
	kmeans.print();

	run_cpu(kmeans);


	return 0;
}
