# PGDDPG


The code of the article "Potential Field Guided Actor-Critic Reinforcement Learning"
Copyright (c) 2020.06. renweiya. email: weiyren.phd@gmail.com. All rights reserved.


tensorflow==1.14.0
gym==0.11.0
python>=3.6

Reward settings (which makes a difficult learning problem):
predator: +10 if all predators catch the prey at the same time. (There are no personal rewards).
prey: +0.1 in each step. (live as long as possible).


Difference
mpe-pgddpg-v-0.1.2.zip: Evaluate the clear action.
mpe-pgddpg-v-0.1.3.zip: Evaluate the noisy action.
(first choice) mpe-pgddpg-v-0.1.4.zip: Evaluate and excute the noisy action.

# type 
python run.py
or
python show_fix_prey.py
