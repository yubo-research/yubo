#!/bin/bash

export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

DDIR=exp_1_k
# f:stybtang-2d f:sphere-2d f:rosenbrock-2d f:ackley-2d f:dixonprice-2d f:griewank-2d f:levy-2d f:michalewicz-2d f:rastrigin-2d f:beale-2d f:branin-2d f:bukin-2d f:crossintray-2d f:dropwave-2d f:eggholder-2d f:hartmann-2d f:holdertable-2d f:shubert-2d f:shekel-2d f:sixhumpcamel-2d f:threehumpcamel-2d

# f:crossintray-2d f:grlee12-2d f:hartmann-3d f:hartmann-4d f:hartmann-6d 
for TAG in f:ackley-10d f:dixonprice-10d f:griewank-10k f:levy-10d f:michalewicz-10d f:rastrigin-10d f:rosenbrock-10d f:sphere-10d f:stybtang-10d
do
# random sobol ioptv_ei ax ei 
 for TTYPE in ioptvp_ei
 # for TTYPE in variance maximin maximin-toroidal iopt
 do
     echo $TAG $TTYPE
     # mkdir -p results/${DDIR}/${TAG}
     python experiments/test.py ${TAG} ${TTYPE} 
 done
done