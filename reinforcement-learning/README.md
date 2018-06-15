## What this is ##

This section contains several reinforcement learning techniques, which is based on [Sung Kim's RL lecture](http://hunkim.github.io/ml/) and his [Github repo](https://github.com/hunkim/ReinforcementZeroToAll).
I used my own implementation of FrozenLake environment(env.py) without using OpenAI to illustrate the entire situation more complehensively.

## Usage ##
The project currently supports two options: interactive mode and learning methods.
```
# python main.py -h
usage: main.py [-h] [-i] [--env_mode {d,s}] [--train {q,dqn}]

optional arguments:
  -h, --help        show this help message and exit
  -i                Enable interactive mode
  --env_mode {d,s}  Whether the environment is stochastic or deterministic
  --train {q,dqn}   Training method
```

Interactive mode shows you what's going on inside the training session.
![preview](images/interactive_mode.png)
