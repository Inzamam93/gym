import gym
import numpy as np
import random
import time
from IPython.display import clear_output

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

print("action space size = ",action_space_size)
print("state space size = ",state_space_size)

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)
num_episodes = 10
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

holesFallen=0
goalsReached=0
# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    print("*****EPISODE ", episode+1, "*****\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        print("\n***STEP#", step, " :\n")
        clear_output(wait=True)
        #env.render()
        #time.sleep(2)
        
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
            print(action);
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        env.render()
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        rewards_current_episode += reward
        
        if done:
            clear_output(wait=True)
            env.render()
            #print("\n")
            if reward == 1:
                print("****You reached the goal!****")
                goalsReached+=1
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                holesFallen+=1         
                time.sleep(3)
            clear_output(wait=True)
            break

        state = new_state
        
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)     

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)
print("Holes fallen",holesFallen)
print("goals reaches", goalsReached)
env.close()
