#!/bin/bash
nvprof --metrics all --csv ./kmeans  -i ../data/data_3d_ind.txt  >  bs_dp_K40c.csv  2>&1
