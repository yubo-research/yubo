#!/bin/bash
cd /home/juju/projects/bbo
export PYTHONPATH=/home/juju/projects/bbo
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3
echo "Start"
for NUM_SAMPLES in 30
    do
    for Q_NUM in 100
        do
        for TAG in  f:dixonprice-1d f:griewank-1d f:levy-1d f:rastrigin-1d f:sphere-1d f:stybtang-1d f:ackley-10d f:dixonprice-10d f:griewank-10d f:levy-10d f:rastrigin-10d f:sphere-10d f:stybtang-10d f:michalewicz-10d f:rosenbrock-10d
            do
            for TTYPE in mtv
                do
                python experiments/exp_3.py ${TAG} ${TTYPE} ${Q_NUM} ${NUM_SAMPLES}
                done
            done
        done
    done
echo "Finish"