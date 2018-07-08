## What this is ##

This section illustrates several reinforcement learning techniques with comments explaining implementation details. Most of the ideas are from [here](#credits--references). I used my own implementation of FrozenLake environment(env.py) instead of OpenAI Gym to illustrate the entire situation more complehensively and add some tweaks to make model converge.

### Milestones ###

- [x] Q-Learning
  - [x] Deterministic / Stochastic
- [x] Deep Q-Learning
  - [x] Deterministic / Stochastic
- [ ] Policy Gradient
  - [x] Deterministic
  - [ ] Stochastic

## Usage ##
```
# python main.py -h
usage: main.py [-h] [-i] [--env_mode {d,s}] [--train {q,dqn,pg}]

optional arguments:
  -h, --help          show this help message and exit
  -i                  Enable interactive mode
  --env_mode {d,s}    Whether the environment is stochastic or deterministic
  --train {q,dqn,pg}  Training method
```

Interactive mode shows you what's going on inside the training session.
![preview](images/interactive_mode.png)

## Credits & References ##
- Sung Kim's RL lecture
  - Lecture: http://hunkim.github.io/ml/
  - Source: https://github.com/hunkim/ReinforcementZeroToAll

- Andrej Karpathy's Implementation of Policy Gradient on Pong
  - Presentation: https://www.youtube.com/watch?v=tqrcjHuNdmQ&t=1403s
  - Explanation Post: http://karpathy.github.io/2016/05/31/rl/
  - Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

## Contributions ##
Contributions in any shape or form will be welcomed 🤘
