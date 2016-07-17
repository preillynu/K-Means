#ifndef _IO_H
#define _IO_H

#include "kmeans.h"

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


#endif
