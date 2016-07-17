#ifndef _KMEANS_H
#define _KMEANS_H

#define MAX_LINE_LEN 4096

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

	void print_param() const;

	void ReadDataFromFile();

	int get_npoints() const;
	int get_nloops() const;
	int get_nfeatures() const;
	float *get_data() const;
	int *get_membership() const;
	//float *get_centers() const;

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

	float 			*data;
	//float 			*centers;
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
	nloops(1000),
	isRMSE(0),
	rmse(0.f),
	isOutput(0)
{
	data = NULL;
	membership = NULL;
	//centers = NULL;
}

Kmeans::~Kmeans() {
	free(data);
	free(membership);
	//free(centers);
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

float* Kmeans::get_data() const
{
	return data; 
}

int* Kmeans::get_membership() const
{
	return membership; 
}

float Kmeans::get_threshold() const
{
	return threshold; 
}

/*
float *Kmeans::get_centers() const
{
	return centers; 
}
*/

void Kmeans::print_param() const
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
void Kmeans::ReadDataFromFile()
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


	//memcpy(features[0], buf, npoints*nfeatures*sizeof(float));

	// allocation 

	membership = (int*) malloc(npoints*sizeof(int));
	for (int i=0; i < npoints; i++) membership[i] = -1;

	printf("\nNumber of objects: %d\n"
			"Number of features: %d\n", npoints, nfeatures);	
}



#endif
