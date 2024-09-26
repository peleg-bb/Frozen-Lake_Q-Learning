import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

environment = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
environment.reset()
q_table = dict.fromkeys([i for i in range(environment.observation_space.n)], [0, 0, 0, 0])

episodes = 100000
alpha = 0.1
gamma = 0.95
epsilon = 0.3


def Train_model():
    """
    This function is used to train the model and plot the graph of episodes vs values of starting state.
    The values of starting state are calculated by taking the average of rewards obtained from the starting state using
    first visit monte carlo method.
    """
    values_of_starting_state = []
    starting_state_visits = 0
    reward_sum = 0
    for episode in tqdm(range(episodes)):
        state = environment.reset()[0]
        done = False
        starting_state_visited = False
        while not done:
            if state == 0 and not starting_state_visited:  # first visit monte carlo
                starting_state_visits += 1
                starting_state_visited = True

            action = get_epsilon_greedy_action(state)
            new_state, reward, terminated, truncated, info = environment.step(action)

            x = update_Q_table(state, action, new_state, reward) # update q_table
            q_table[state] = [x if i == action else q_table[state][i] for i in range(environment.action_space.n)]

            state = new_state
            done = terminated or truncated

            if reward:  # sum value function of starting state
                reward_sum += reward

        if episode % 10000 == 0:
            values_of_starting_state.append(reward_sum / starting_state_visits)

    plot_graph(values_of_starting_state)


def plot_graph(values_of_starting_state):
    """
    This function is used to plot the graph of episodes vs values of starting state.
    :param values_of_starting_state:
    :return:
    """
    episodes_range = list(range(0, episodes, 10000))
    plt.plot(episodes_range, values_of_starting_state)
    plt.xlabel('Episodes')
    plt.ylabel('Value of Starting State')
    plt.title('Episodes vs Values of Starting State')
    plt.show()


def get_epsilon_greedy_action(state):
    """
    This function is used to get the action using epsilon greedy policy
    """
    rnd = np.random.random()
    if rnd >= epsilon:
        action = np.argmax(q_table[state])
    else:
        action = environment.action_space.sample()
    return action


def update_Q_table(state, action, new_state, reward):
    return (q_table[state][action] + alpha *
            (reward + gamma * np.max(q_table[new_state]) - q_table[state][action]))


Train_model()
print(q_table)

