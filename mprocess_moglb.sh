#!/bin/bash
sinit=200
outerloop=10
innerloop=10
for i in $(seq 1 $outerloop)
do
    # echo $i
    for j in $(seq 1 $innerloop)
    do
        let "cInd = $j-1 + ($i-1) * $innerloop"
        python MOGLB.py $1 $2 $3 $4 $5 --seed $sinit --idProcess $cInd &
    done
    wait $!
    echo $i
done
