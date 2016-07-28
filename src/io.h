#ifndef _IO_H_
#define _IO_H_

#include <stdio.h>                                                              
#include <stdlib.h>                                                             
#include <iostream>  
#include <time.h>                                                               
#include <sys/time.h> 

// cpu timer                                                                
struct timeval startTime, endTime;                                          

void cputic() {                                              
	gettimeofday(&startTime, NULL);                                         
}                                                                           

void cputoc() {                                                
	gettimeofday(&endTime, NULL);                                           
}                                                                           

void printCpuTime() {                                              
	long seconds, useconds;                                                 
	seconds  = endTime.tv_sec  - startTime.tv_sec;                          
	useconds = endTime.tv_usec - startTime.tv_usec;                         
	double mtime = useconds;                                                
	mtime/=1000;                                                            
	mtime+=seconds*1000;                                                    
	printf("CPU Elapsed Time : %f ms\n", mtime);                            
}                        

class PARAMS {
public:
	PARAMS() {
		filename = NULL;	
		threshold = 0.001f;
		max_nclusters = 5;
		min_nclusters = 5;
		nloops = 1000;
		isRMSE = 0;
		isOutput = 0;
	}
	~PARAMS() {}

	char            *filename;                                                  
	float           threshold;                                                  
	int             max_nclusters;                                              
	int             min_nclusters;                                              
	int             nloops;                                                     
	int             isRMSE;                                                     
	int             isOutput;                                                   
};

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

void readcmdline(int argc, char **argv, PARAMS &params)
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
				params.filename = (optarg);
				break;
			case 't': 
				params.threshold = atof(optarg);
				break;
			case 'm': 
				params.max_nclusters = atoi(optarg);
				break;
			case 'n': 
				params.min_nclusters = atoi(optarg);
				break;
			case 'l': 
				params.nloops = atoi(optarg);
				break;
			case 'r':
				params.isRMSE = 1;
				break;
			case 'o': 
				params.isOutput = 1;
				break;
			case 'h': 
				print_usage(argv[0]);
				exit(1);
			case '?': 
				print_usage(argv[0]);
				exit(1);
		}
	}


	if(params.filename == NULL) {
		print_usage(argv[0]);
		exit(1);
	}

}


#endif
