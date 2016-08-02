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
#include "kmeans_gpu.h"
//#include "kmeans.h"

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
	//kmeans_cpu.print_param();

	/// benchmark the runtime on cpu
	cputic();
	run_cpu(kmeans_cpu);
	cputoc();
	printCpuTime();

	//------------------------------------------------------------------------//
	// run gpu version
	//------------------------------------------------------------------------//
	KmeansGPU kmeans_gpu;

	kmeans_gpu.filename      = params.filename;
	kmeans_gpu.threshold     = params.threshold;
	kmeans_gpu.max_nclusters = params.max_nclusters;
	kmeans_gpu.min_nclusters = params.min_nclusters;
	kmeans_gpu.nloops        = params.nloops;
	kmeans_gpu.isRMSE        = params.isRMSE;
	kmeans_gpu.isOutput      = params.isOutput;

	kmeans_gpu.ReadDataFromFile();

	/// 
	kmeans_gpu.print_param();

	// run gpu version
	kmeans_gpu.runKmeans();

	return 0;
}
