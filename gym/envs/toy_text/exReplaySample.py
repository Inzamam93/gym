import gym
import os
import random
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

microtime = lambda: int(round(time.time() * 1000))
start_t = microtime()

class ExperienceBuffer():
    def __init__(self, buf_size):
        self.buf_size = buf_size

        self.buffer = []

    # Store experiences
    def add(self, experience):
        if len(self.buffer) + 1 >= self.buf_size:
            self.buffer.pop(0)

        self.buffer.append(experience)

    # Retrtieve a random sample of experiences
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        
        return np.array(samples)

class QNetwork():
    def __init__(self, num_states, num_actions, save_file=None, gamma=0.99, lr=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.save_file = save_file
        self.gamma = gamma
        self.lr = lr

        if save_file is not None:
            try:
                load = pickle.load(open(save_file, "rb"))
                self.W = tf.Variable(load)
                
                print("Loaded %s"  % save_file)
                print(load)
                print("")
            except FileNotFoundError:
                self.W = tf.Variable(tf.random_uniform(
                    [self.num_states, self.num_actions],
                    0,
                    0.01),
                dtype=tf.float32)
            except Exception as _e:
                print(_e)
        else:
            self.W = tf.Variable(tf.random_uniform([self.num_states, self.num_actions], 0, 0.01), dtype=tf.float32)

        self.input_state = tf.placeholder(shape=[None], dtype=tf.int32, name="input_state")
        self.input_state_one_hot = tf.one_hot(
            indices=tf.cast(self.input_state, tf.int32),
            depth=self.num_states
        )
        self.Q = tf.matmul(self.input_state_one_hot, self.W)
        self.Q_target = tf.placeholder(
            shape=[None, self.num_actions],
            dtype=tf.float32,
            name="Q_target"
        )
        self.best_action = tf.argmax(self.Q, 1)

        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q), 1)
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.train_op = self.trainer.minimize(self.loss)

    def save(self, val):
        if self.save_file is None:
            return
        
        pickle.dump(val, open(self.save_file, "wb"))
        
# Setup
train = True
batch_train = True
test = True

pre_train_steps = 5000
train_freq = 1

num_episodes = 1000
num_episodes_test = 100
num_steps = 100

e_start = 0.1
e_end = 0.001

#QN1 = QNetwork(16, 4, save_file="FrozenLake-v0.p", gamma=0.99, lr=0.1)
QN1 = QNetwork(16, 4, gamma=0.99, lr=0.1)

# Variables
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')
#env = gym.make("FrozenLake-v0")
env = gym.wrappers.Monitor(env, "tmp/FrozenLake-0.1", force=True)
exp_buf = ExperienceBuffer(1000)

e_factor = ((e_start - e_end) / (num_episodes * (num_steps / 5)))
e = e_start

bench = [[], [], [], [], []]

# Add an operation to initialize global variables.
init_op = tf.global_variables_initializer()

# Training
with tf.Session() as sess:
    sess.run(init_op)
    
    if train == True:
        print("Training started\n")

        batch_training_started = False
        total_batch_trained = 0
        all_rewards = []
        all_steps = []
        total_steps = 0
        
        for episode in range(num_episodes):
            if episode % 100 == 0 and episode != 0:
                t = microtime()
                W_val = sess.run(QN1.W)
                QN1.save(W_val)
                
                print("Episodes %04d - %04d: %i succeeded, %.2f avg steps/episode, e=%.4f" % (
                        episode - 100,
                        episode,
                        sum(all_rewards[-100:]),
                        np.mean(all_steps[-100:]),
                        e
                    )
                )
                bench[0].append((microtime() - t))

            # Reset episode-specific parameters
            state = env.reset()
            steps = 0
            episode_reward = 0
            done = False

            # Do steps in the game
            while steps <= num_steps:
                if done == True:
                    #if reward==1:
                        #print("***Goal Reached***")
                        #time.sleep(3)
                    break

                t = microtime()
                # An e chance of randomly selection an action
                if (np.random.rand(1) < e) or (total_steps < pre_train_steps):
                    act = env.action_space.sample()
                else:
                    # Obtain the best action and current Q_values for this state
                    act = sess.run(QN1.best_action, feed_dict={
                        QN1.input_state: [state]
                    })
                    act = act[0]
                bench[1].append((microtime() - t))

                # Advance a state
                t = microtime()
                new_state, reward, done, _ = env.step(act)
                bench[2].append((microtime() - t))

                #My addition - Render the environment
                #env.render()
                
                # Decrease the random % for every action
                e -= e_factor

                if e < e_end:
                    e = e_end

                # Store this experience
                exp_buf.add((state, act, reward, new_state, done))

                # Train from memory
                if (batch_train == True) and (total_steps > pre_train_steps) and ((total_steps % train_freq) == 0):
                    if batch_training_started == False:
                        batch_training_started = True
                        print("Batch training started")
                        
                    training_batch = exp_buf.sample(16)

                    t = microtime()
                    batch_new_Qs = sess.run(QN1.Q, feed_dict={
                        QN1.input_state: training_batch[:,3]
                    }) # Q(s', a')

                    batch_curr_Qs = sess.run(QN1.Q, feed_dict={
                        QN1.input_state: training_batch[:,0]
                    }) # Q(s, a)
                    bench[3].append((microtime() - t))
                    
                    # Best possible outcome of the new states (per state)
                    new_Qs_max = np.max(batch_new_Qs, 1) # max a' for Q(s', a')

                    target_Qs = batch_curr_Qs.copy()
                    for i, experience in enumerate(training_batch):
                        s, a, r, ss, d = experience # s a r s' d
                        
                        target_Qs[i][int(a)] = r + QN1.gamma * new_Qs_max[i]
                    # target for a = r + y*maxa'Q(s', a')

                    # Train with the given state(s) and target_Qs
                    t = microtime()
                    sess.run(QN1.train_op, feed_dict={
                        QN1.input_state: training_batch[:,0],
                        QN1.Q_target: target_Qs
                    }) # train with target and s
                    bench[4].append((microtime() - t))

                    total_batch_trained += len(training_batch)

                steps += 1
                total_steps += 1
                episode_reward += reward
   
                state = new_state

            all_rewards.append(episode_reward)
            all_steps.append(steps)

        W_val = sess.run(QN1.W)
        QN1.save(W_val)

        print("\nCompleted %i organic steps" % sum(all_steps))
        print("Completed %i batch-trained steps" % total_batch_trained)

    if test == True:
        # Testing
        print("\nTesting...")
        
        all_rewards = []
        all_steps = []
        
        for episode in range(num_episodes_test):
            # Reset episode-specific parameters
            state = env.reset()
            steps = 0
            episode_reward = 0
            done = False

            # Do steps in the game
            while steps <= num_steps:
                if done == True:
                    break

                act = sess.run(QN1.best_action, feed_dict={
                    QN1.input_state: [state]
                })
                act = act[0]

                new_state, reward, done, _ = env.step(act)

                steps += 1
                episode_reward += reward
                state = new_state

            all_rewards.append(episode_reward)
            all_steps.append(steps)

        print("Finished. %i/%i succeeded, avg. steps %.2f" % (
            sum(all_rewards),
            num_episodes_test,
            np.mean(all_steps)
        ))

    print("\nQ network:")
    W_val = sess.run(QN1.W)
    print("\n  left       down       right      up")
    print(W_val)

print("\nTimes:\nsave, get_act, step, get_new_Qs, train:")
print(", ".join([str(sum(t)) for t in bench]))

print("\nTotal took %i ms" % (microtime() - start_t))
env.close()
