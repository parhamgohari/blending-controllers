# Blending Controllers via Multi-Objective Bandits
This package blends controllers that optimize different objective functions.

In the SafetyGym framework, we blend a safe and a performant controller. The safe controller minimzes the possible collisions with obstacles while the performant controller cares only about efficiently maximize a reward signal (encoding the goal of reaching a target or pushing and object towards a target).

## Supported Platforms
This package has been tested on Ubuntu 16.04 LTS, and it is probably fine for most recent Mac and Linux operating systems.

Requires **Python 3.6 or greater**.

## REQUIRED: Mujoco

1. Obtain a 30-day free trial on the [MuJoCo website](https://www.roboti.us/license.html)
   or free license if you are a student.
   The license key will arrive in an email with your username and password.
2. Download the MuJoCo version 2.0 binaries for
   [Linux](https://www.roboti.us/download/mujoco200_linux.zip) or
   [OSX](https://www.roboti.us/download/mujoco200_macos.zip).
3. Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`,
   and place your license key (the `mjkey.txt` file from your email)
   at `~/.mujoco/mjkey.txt`.


## Installation
First install the package `tikzplotlib` for saving .tex file
```
pip install tikzplotlib
```

Clone the package and install all the submodules
```
git clone https://github.com/parhamgohari/blending-controllers.git

cd blending-controllers

git submodule update --init --recursive
```

### Install SafetyGym

This package needs [SafetyGym](https://openai.com/blog/safety-gym/) to run simulations. We provide a custom modification of 
SafetyGym with this repository.
```
git clone https://github.com/openai/safety-gym.git

cd blending-controllers/safety-gym

python -m pip install -e .
```

### Install safety-starter-agents

Install our slightly modified version of safety-starter-agents.
```
cd blending-controllers/safety-starter-agents

python -m pip install -e .
```

## Learning and testing a policy

### Learning a safe or a performant controller

To learn a performant or a safe controller, we have to generate and save the policies through

```
cd blending-controllers/safety-starter-agents/scripts

python experiment.py --algo ALGO --task TASK --robot ROBOT --seed SEED
    --exp_name EXP_NAME --cpu CPU --epochs EPOCHS --steps_per_epoch STEP
```

where the arguments are optional and given below

* `ALGO` is in `['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']`.
* `TASK` is in `['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']` .
* `ROBOT` is in `['point', 'car', 'doggo']`.
* `SEED` is an integer. In the paper experiments, we used seeds of 0, 10, and 20, but results may not reproduce perfectly deterministically across machines.
* `CPU` is an integer for how many CPUs to parallelize across.
* `EPOCHS` is the number of epochs to use for the simulation.
* `STEP` is the number of steps per epochs for the simulation.
* `EXP_NAME` is an argument for the name of the file where results will be saved. The save folder will be placed in `blending-controllers/safety-starter-agents/data`.

For example, the safe and the performant controller for the `CarButton` environment can be obtained by
```
cd blending-controllers/safety-starter-agents/scripts

# The performant controller will be save in blending-controllers/safety-starter-agents/data/__date__performant_car_button1
python experiment.py --algo trpo --task button1 --robot car --exp_name performant_car_button1 --cpu 6 --epochs 334 --steps_per_epoch 30000

# The safe controller will be save in blending-controllers/safety-starter-agents/data/__date__save_car_button1
python experiment.py --algo cpo --task button1 --robot car --exp_name safe_car_button1 --cpu 6 --epochs 334 --steps_per_epoch 30000
```

### Blending a safe and a performant controller

```
cd blending-controllers/

python blending_algo.py --fpath_controllers path_perf path_safe --seed SEED --save_file filename --idProcess --num_env_interact envInteract 
						--steps_per_epoch STEP --max_ep_len len 
```

where, except for `--fpath_controllers` and `--save_file`, the arguments are optional and given below

* `path_perf` is the performant controller in `blending-controllers/safety-starter-agents/data/*`.
* `path_safe` is the safe controller in `blending-controllers/safety-starter-agents/data/*`.
* `SEED` is an integer. In the paper experiments, we mainly used seed ```201``` and its increments for multiple traces.
* `filename` is the output file name without any extension. This will be saved in the current directory.
* `envInteract` is the total number of environment interaction (called `T` in the paper). Default value is `3.3M`/
* `STEP` is the number of steps per epochs for the simulation. Default value is `30000`.
* `max_ep_len` is the maximum number of environment interacts for an episode. Default value is `1000`.

For example, To blend the performant and the safe policy obtained from the `CarButton` example above, run

```
cd blending-controllers/

python blending_algo.py --fpath_controllers safety-starter-agents/data/__date__performant_car_button1/__date__performant_car_button1_ safety-starter-agents/data/__date__safe_car_button1/__date__performant_car_button1_ --seed 201 --save_file blend_car_button1 --idProcess 0
```

### Testing and saving the metrics for the safe and the performant controller

For testing the performant or the save controller and computing the metrics in the same environment as for the blending, the seed value
has to be the same as for the blending algorithm as below
```
cd blending-controllers/

# For the performant controller above
python test_policy.py safety-starter-agents/data/__date__performant_car_button1/__date__performant_car_button_1_ --seed 201 --save_file performant_car_button1 --idProcess 0

# For the safe controller above
python test_policy.py safety-starter-agents/data/__date__safe_car_button1/__date__safe_car_button_1_ --seed 201 --save_file safe_car_button1 --idProcess 0
```

### Compare the blended, the performant, and the safe controller 

To compare the different controllers, we propose `plot_result.py` that can be launched using the save_file as below
```
cd blending-controllers/

# For the blended controller obtained above
python plot_result.py --logdir blend_car_button1 performant_car_button1 safe_car_button1 --legend 'Blended controller' 'Performant controller' 'Safe controller' --colors red blue green --num_traces 1 1 1 --ind_traces 0 0 0 --steps_per_epoch 30000 --window 10 --output_name car_button
```

To plot the learning curves of the safe and the performant controller, execute
```
cd blending-controllers/safety-starter-agents/scripts/

# For example the performant controller
python plot.py ../data/__date__performant_car_button1/__date__performant_car_button_1_ --legend Performant --value AverageEpRet AverageEpCost
```