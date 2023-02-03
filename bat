#!/bin/bash

export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

DDIR=exp_1_h
for TAG in f:sphere-2d f:sphere-3d f:sphere-10d f:sphere-30d
# for TAG in mcc lunar ant
#for TAG in lunar mcc f:sphere-2d # f:sphere-3d f:sphere-10d mcc lunar
do
 for TTYPE in iei # iopt # random sobol ei ucb ax idopt
 # for TTYPE in variance maximin maximin-toroidal iopt
 do
     echo $TAG $TTYPE
     mkdir -p results/${DDIR}/${TAG}
     python experiments/exp_1.py ${TAG} ${TTYPE} &> results/${DDIR}/${TAG}/${TTYPE} &
 done
done
