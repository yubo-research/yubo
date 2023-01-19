#!/bin/bash

export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

DDIR=exp_1_c
for TAG in mcc lunar ant f:sphere-2d
do
 # for TTYPE in sobol random dumb
 for TTYPE in params actions-corr # actions bayes-actions bayes-params
 do
     echo $TAG $TTYPE
     mkdir -p results/${DDIR}/${TAG}
     python experiments/exp_1.py ${TAG} ${TTYPE} &> results/${DDIR}/${TAG}/${TTYPE} &
 done
done
