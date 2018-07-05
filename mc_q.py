import gym
import math
import numpy as np
import copy

# Initialize environment
env = gym.make('MountainCar-v0')

# env.reset()
# for i in range(1000):
# 	env.render()
# 	obs,rew,done,_ = env.step(env.action_space.sample())
# 	print(obs)

# Greedy parameter
epsilon_max = 1
epsilon_min = .1
# epsilon = .1

# Discount factor
gamma = .99

# Learning rate
alpha_max = 1
alpha_min = .05
# alpha = .05

# Set action parameters
ACTION_SIZE = env.action_space.n

# Set state space parameters
DISC_POS_SIZE = 10
DISC_V_SIZE = 30

# Fixed values
STATE_SIZE = 2
BUCKETS = (DISC_POS_SIZE, DISC_V_SIZE)
CUM_PROD = [1, np.prod(BUCKETS[0:1])]
SPACE_SIZE = ACTION_SIZE*DISC_POS_SIZE*DISC_V_SIZE
ACTION_SPACE_SIZE = DISC_POS_SIZE*DISC_V_SIZE

# Get bounds of state space
# STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS = [[-1.2, .6], [-.07,.07]]

#####			      		#####
#####		Q-Function		#####
##### 						#####
# Takes an interval and discretizes it into 'size' buckets
def discretized_state_bucket(interval, size, value):
	interval_length = (interval[1]+0.0)/size - (interval[0]+0.0)/size
	return int(np.floor(value/interval_length - (0.0+interval[0])/interval_length))

# Given an action and state vector, returns corresponding index of q function
def get_q_index(action, state_array):
	index = action*SPACE_SIZE/ACTION_SIZE
	for i in range(STATE_SIZE):
		bucket = discretized_state_bucket(STATE_BOUNDS[i],BUCKETS[i],state_array[i])
		index += CUM_PROD[i]*bucket
	return index


def get_q_action(state):
	zero_index = get_q_index(0, state)
	one_index = zero_index + ACTION_SPACE_SIZE#get_q_index(1, state)
	two_index = one_index + ACTION_SPACE_SIZE#get_q_index(2, state)
	Q_arr = [Q[zero_index], Q[one_index], Q[two_index]]
	return np.argmax(Q_arr)

def get_max_q_value(state):
	zero_index = get_q_index(0, state)
	one_index = zero_index + ACTION_SPACE_SIZE#get_q_index(1, state)
	two_index = one_index + ACTION_SPACE_SIZE#get_q_index(2, state)
	Q_arr = [Q[zero_index], Q[one_index], Q[two_index]]
	return np.amax(Q_arr)

def get_action(state, eps):
	if (np.random.uniform() < eps):
		return np.random.randint(0,3)
	else:
		return get_q_action(state)

# Computes ||Q_{i+1} - Q_i||_2
def compute_qnorm(Q_old):
	return np.linalg.norm(np.subtract(Q_old,Q), 2)


# Initialize q function
# Q = np.random.random([SPACE_SIZE])
Q = np.zeros(SPACE_SIZE)

num_episodes = 15000
eps_decay_rate = (epsilon_min - epsilon_max + 0.0)/num_episodes
alpha_decay_rate = (alpha_min - alpha_max + 0.0)/num_episodes
q_norm_plot_array = np.zeros(num_episodes)
q_avg_plot_array = np.zeros(num_episodes)
reward_plot_array = np.zeros(num_episodes)

for i in range(num_episodes):
	# Initialize state
	# [Position, Velocity, Angle, Angular velocity]
	S = env.reset()
	done = False
	env.reset()
	Q_old = copy.copy(Q)
	tot_reward = 0
	while not done:
		if (i < 5000):
			action = np.random.randint(0,3)
		else:
			# action = get_action(S, epsilon_max + eps_decay_rate*i)
			action = get_action(S, np.maximum(.1, np.minimum(1, 2.5-np.log(i+1))))
		# action = get_action(S, epsilon)
		index = get_q_index(action, S)
		new_state, reward, done, _ = env.step(action)
		# alpha = alpha_max + alpha_decay_rate*i
		alpha = np.maximum(.1, np.minimum(1, 2.5-np.log(i+1)))
		Q[index] += alpha*(reward + gamma*get_max_q_value(new_state) - Q[index])
		S = new_state
		tot_reward += reward
		if (i > (num_episodes - 10)):
			env.render()
		tot_reward += reward
	reward_plot_array[i] = tot_reward
	q_norm_plot_array[i] = compute_qnorm(Q_old)
	q_avg_plot_array[i] = np.mean(Q)


np.savetxt('norm_values.csv', q_norm_plot_array, delimiter=',')
np.savetxt('avg_values.csv', q_avg_plot_array, delimiter=',')
np.savetxt('rewards.csv', q_avg_plot_array, delimiter=',')
np.savetxt('q_values.csv', Q, delimiter=',')