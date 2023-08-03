#!/bin/bash
cd /home/juju/projects/bbo
export PYTHONPATH=/home/juju/projects/bbo
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3

for TAG in f:sphere-4d
# f:branin-2d f:ackley-2d f:sphere-30d f:levy-2d f:levy-10d f:levy-30d f:stybtang-2d f:stybtang-10d f:stybtang-30d f:rosenbrock-2d f:rosenbrock-10d f:rosenbrock-30d f:dixonprice-2d f:dixonprice-10d f:dixonprice-30d f:griewank-2d f:griewank-10d f:griewank-30d f:michalewicz-2d f:michalewicz-10d f:michalewicz-30d f:rastrigin-2d f:rastrigin-10d f:rastrigin-30d f:beale-2d f:bukin-2d f:crossintray-2d f:dropwave-2d f:eggholder-2d f:hartmann-2d f:holdertable-2d f:shubert-2d f:shekel-4d f:sixhumpcamel-2d f:threehumpcamel-2d f:grlee12-1d f:hartmann-3d f:hartmann-4d f:hartmann-6d
do 
#  random sobol ieig sobol_mes sobol_gibbon iopt_ei pes ieig_znm ieig_cem ieig_ts cem_ts sr
 for TTYPE in random sobol
 # for TTYPE in variance maximin maximin-toroidal iopt
 do
     echo "Start" $TAG $TTYPE
     # mkdir -p results/${DDIR}/${TAG}
     python experiments/run_result1.py ${TAG} ${TTYPE} >/dev/null 2>&1
     echo "Finish"
 done
done