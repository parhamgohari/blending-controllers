#!/bin/bash

# The initial seed --> Each process have a seed of sinit + idProcess
sinit=201

# Outer loop specifying the number of time innerloop processor must be launch
# Note that outerloop*innerloop gives the number of traces
outerloop=1

# Number of processes to used at the same time
innerloop=1

for i in $(seq 1 $outerloop)
do
    for j in $(seq 1 $innerloop)
    do
        let "cInd = 5 + j-1 + ($i-1) * $innerloop"
        python blending_algo.py $1 $2 $3 $4 $5 --seed $sinit --idProcess $cInd &
    done
    wait $!
    echo $i
done
