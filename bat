#!/bin/bash

export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

DDIR=exp_1_f
for TAG in f:sphere-2d f:sphere-3d f:sphere-10d f:sphere-30d
# for TAG in mcc lunar ant # mcc
do
 #for TTYPE in random sobol dumb
 for TTYPE in variance maximin maximin-toroidal iopt
 do
     echo $TAG $TTYPE
     mkdir -p results/${DDIR}/${TAG}
     python experiments/exp_1.py ${TAG} ${TTYPE} &> results/${DDIR}/${TAG}/${TTYPE} &
 done
done
