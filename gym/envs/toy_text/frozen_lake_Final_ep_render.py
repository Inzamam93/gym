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

num_episodes = 4
max_steps_per_episode = 30

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.005

rewards_all_episodes = []
goal_reached=0
holesEncountered=0
succesfulEpisode=0


# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    #print("*****EPISODE ", episode+1, "*****\n")
    #time.sleep(1)

    episode+1
    for step in range(max_steps_per_episode):
       
        #env.render()
        
        
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) 
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        if episode==3:
            print("\n\nrandomly generated exploration rate threshold =",exploration_rate_threshold)
            print("\nStep #",step)
            time.sleep(1)
            env.render()
            
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward
        
        if done:
            if reward == 1:
                goal_reached+=1
                succesfulEpisode=episode
            else:
                holesEncountered+=1
            break

        
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)     

env.close()
print("\nLast exploration rate = ",exploration_rate)
# Print updated Q-table
print("\n\n********Q-table********\n")
print(" Left        Down        Right      Up\n")
print(q_table)
print("Number of times reached goal =",goal_reached)
print("Number of holes encountered =",holesEncountered)
print("Goal last reached during episode : ", succesfulEpisode)

