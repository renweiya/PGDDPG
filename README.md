### PGDDPG


###### The code of the article "Potential Field Guided Actor-Critic Reinforcement Learning" (PGDDPG)  
###### Copyright (c) 2020.06. renweiya. email: weiyren.phd@gmail.com. All rights reserved. 

### requires: 
###### tensorflow==1.14.0, 
###### gym==0.11.0, 
###### python>=3.6.

### Reward settings (which makes a difficult learning problem): 
###### predator: +10 if all predators catch the prey at the same time. (There are no personal rewards).  
###### prey: +0.1 in each step. (live as long as possible).  


### (MA)DDPG fails without reward shaping. You need a well designed reward shaping if use DDPG or MADDPG.


### test PGDDPG files:

###### run_fix_prey.py: train predators with a well trained prey named prey-23.
###### run_fix_preys.py: train predators with a well trained prey named prey-s.
###### run.py: train predators and prey simutaniously.
