# Improving sample efficiency of high dimensional Bayesian optimization with MCMC on approximated posterior ratio


## Installation

There is `requirements.txt` in this repository, which are mandatory to run MCMC-BO on synthetic functions.

If you want to run experiment on Mujoco locomotion tasks, please refer to https://github.com/openai/mujoco-py to configure corresponding environment.

## How to run the algorithm

You can test MCMC-BO by running the following command:

```bash
python exp.py --func Ackley --dim 200  --tr_num 1 --eval_num 6000 --init_num 200 --batch_size 100 --noise_var 0 --repeat_num 30 --use_mcmc 1 --gpu_idx 0
```
