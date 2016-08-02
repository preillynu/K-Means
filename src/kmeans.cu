/*
   The interface is using kmeans in rodinia 3.0 benchmarks.
*/
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <math.h>
#include <iostream>

#include <cuda_runtime.h>                                                       
#include <helper_cuda.h> 

#include "io.h"
#include "kmeans_cpu.h"
//#include "kmeans.h"
//#include "kmeans_gpu.h"

int main(int argc, char **argv)
{
	//------------------------------------------------------------------------//
	// read command line 
	//------------------------------------------------------------------------//
	PARAMS params;
	readcmdline(argc, argv, params);

	//------------------------------------------------------------------------//
	// run cpu version
	//------------------------------------------------------------------------//
	KmeansCpu kmeans_cpu;
	/// pass parameters
	kmeans_cpu.filename      = params.filename;
	kmeans_cpu.threshold     = params.threshold;
	kmeans_cpu.max_nclusters = params.max_nclusters;
	kmeans_cpu.min_nclusters = params.min_nclusters;
	kmeans_cpu.nloops        = params.nloops;
	kmeans_cpu.isRMSE        = params.isRMSE;
	kmeans_cpu.isOutput      = params.isOutput;
	/// read input data from the specified file
	kmeans_cpu.ReadDataFromFile();
	/// check the kmeans configurations
	kmeans_cpu.print_param();
	/// benchmark the runtime on cpu
	cputic();
	run_cpu(kmeans_cpu);
	cputoc();
	printCpuTime();





	//Kmeans kmeans_gpu;

	/*
	readcmdline(argc, argv, kmeans_gpu);

	kmeans_gpu.ReadDataFromFile();

	kmeans_gpu.print_param();


	//--------------//
	// run gpu version
	//--------------//
	run_gpu(kmeans_gpu);
*/
	return 0;
}
