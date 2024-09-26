# Frozen-Lake_Q-Learning
In this project I implemented Gymnasium's Frozen Lake Problem - https://gymnasium.farama.org/environments/toy_text/frozen_lake/
We used agent-based reinforcement learning to train the model to solve the Frozen Lake maze.
The model uses a Q table which is updated using the formula- Q(s,a)_(t+1)= Q(s,a)_t+α(R+γQ(S_(t+1),A^' )-Q(S_t,A_t)). 

This was part of the Reinforcement Learning course, a voluntary master's course I took as part of my Bachelor's of Software Engineering at Ben Gurion University.
