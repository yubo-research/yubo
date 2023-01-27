#!/bin/bash

export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

DDIR=exp_1_f
# for TAG in f:sphere-2d f:sphere-3d f:sphere-10d
for TAG in f:sphere-30d f:sphere-100d
# for TAG in mcc lunar ant # mcc
do
 for TTYPE in iopt params-toroidal sobol random # actions # actions-corr params-toroidal sobol random # params
 # for TTYPE in params actions-corr # actions bayes-actions bayes-params
 do
     echo $TAG $TTYPE
     mkdir -p results/${DDIR}/${TAG}
     python experiments/exp_1.py ${TAG} ${TTYPE} &> results/${DDIR}/${TAG}/${TTYPE} &
 done
done
