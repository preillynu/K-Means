#!/bin/bash
nvprof --metrics all --csv ./kmeans  -i ../data/data_3d_ind.txt  >  baseline_k40c.csv  2>&1
