# blending-controllers
Blending a safe and a performant controller
## Supported Platforms
This package has been tested on Mac OS Catalina and Ubuntu 16.04 LTS, and is probably fine for most recent Mac and Linux operating systems.
Requires **Python 3.6 or greater.**
## Installation
Our code uses Safety Gym (https://github.com/openai/safety-gym) to run simulations. Safety Gym requires mujoco_py, so the first step would be installing mujoco_py (https://github.com/openai/mujoco-py) and then installing Safety Gym. Note that mujoco_py **requires Python 3.6 or greater**, so Safety Gym does as well.
Afterwards, install using pip install.

### We furthermore need GUROBI for solving a QCQP (https://www.gurobi.com/) optimization problem in the blending algorithm.
```
git clone https://github.com/parhamgohari/blending-controllers.git
cd blending-controllers
git submodule update --init --recursive
cd safety-starter-agents
pip install -e .
cd ../
```
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
```
