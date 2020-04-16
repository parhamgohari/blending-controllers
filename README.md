# blending-controllers
This package blends controllers that optimize different objectives functions.

In the SafetyGym framework, this cooresponds to blending a safe and a performant controller. The safe controller minimzes the possible collisions with obstacles while the performant controller cares only about reaching the goal.

## Supported Platforms
This package has been tested on Mac OS Catalina and Ubuntu 16.04 LTS, and it is probably fine for most recent Mac and Linux operating systems.

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

Clone the package and install all the submodules
```
git clone https://github.com/parhamgohari/blending-controllers.git

cd blending-controllers

git submodule update --init --recursive
```

### Optional but recommended
We recommend to install **conda** and use the virtual environement that contains the dependencies needed for this package.
- For MacOS, install Anaconda using [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
- For Linux, install Andaconda using [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Install the conda virtual environment from blending-controllers.
```
cd blending-controllers

conda env create --file blending.yml

conda activate blending
```

**Every instructions below need to be executed in the blending environment**.


### Install SafetyGym

This package needs [SafetyGym](https://openai.com/blog/safety-gym/) to run simulations. Clone and install SafeyGym framework **while the conda blending environment is activated**.
```
git clone https://github.com/openai/safety-gym.git

cd safety-gym

python -m pip install -e .
```

### Install safety-starter-agents

Install our slightly modified version of safety-starter-agents.
```
cd blending-controllers/safety-starter-agents

python -m pip install -e .
```
<!--
## Learning a safe or performant policy
```
cd blending-controllers/safety-starter-agents/scripts
python experiment.py --cpu=cpu --objective='performant or safe' --task=Goal2 --seed=seed
```
where cpu specifies the number of cores you want to allocate and seed is a seed for the gym simulator. The choice of the performant or safe policy can be specified using --objective.

## Testing a learnt policy
To test a policy, check if the policy has been first created in the '''blending-controllers/safety-starter-agents/data'''. Then execute the following command
```
cd blending-controllers/safety-starter-agents/scripts
python test_policy.py /path/to/data
```

## Blending policy
To run the blending algorthm, follow the commands
```
cd blending-controllers/
python3 MOGLB.py <directory to the safe policy> <directory to the performant policy>
```

## Visualizing the plots
To show the plots as presented in the paper, a safe and performant policies must have been first computed with a blended policy. Then, execute the following command
```
cd blending-controllers/safety-starter-agents/scripts
python3 plot.py <data CPO> <data PPO-Lagragian> --safepath <directory to the safe policy> --perfpath <directory to the performant policy> --blendpath <path> --regretpath <path> --legend CPO PPO-Lagragian --value AverageEpRet AverageEpCost
```
``` -->
