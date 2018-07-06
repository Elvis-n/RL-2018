import gym
import math
import numpy as np
import copy
from matplotlib import pyplot as plt

def vec_product(v): 
	prod = 1
	for i in v:
		prod *= i
	return prod

def get_cum_prod(v):
	cum_prod = np.zeros_like(v)
	for i in range(np.shape(v)[0]):
		prod = 1
		for j in range(i):
			prod *= v[j]
		cum_prod[i] = prod
	return cum_prod

# Takes an interval and discretizes it into 'size' buckets  
def discretized_state_bucket(interval, size, value):
	interval_length = (interval[1]+0.0)/size - (interval[0]+0.0)/size
	return int(np.floor(value/interval_length - (0.0+interval[0])/interval_length))

class Playground:

	# Initializes Playground class
	# Parameters:
	# - game: A string denoting name of game
	#		  Example: 'MountainCar-v0'
	# - buckets: A vector where each entry denotes size of each action discretization
	# 			 Example: [10,30]
	# - bounds: A vector of vector pairs denoting bounds for each action 
	#           Example: [[-1.2,.6], [-.07, .07]]
	# - epsilon: A number in [0,1] denoting greedy probability parameter
	#            Example: .3
	# - alpha: A number in [0,1] denoting learning rate
	#          Example: .05
	# - gamma: A number in [0,1] denoting discount factor
	#		   Example: .99
	def __init__(self, game, buckets, bounds, epsilon_max, epsilon_min, alpha, gamma):
		self.game = game
		self.buckets = buckets
		self.bounds = bounds
		self.epsilon_max = epsilon_max
		self.epsilon_min = epsilon_min
		self.alpha = alpha
		self.gamma = gamma

	def get_q_index(self, action, state_array):
		index = action*self.ACTION_SPACE_SIZE
		for i in range(self.STATE_SIZE):
			bucket = discretized_state_bucket(self.bounds[i],self.buckets[i],state_array[i])
			index += self.CUM_PROD[i]*bucket
		return index

	def read_q(self, state, action):
		zero_index = self.get_q_index(0, state)
		Q_arr = np.zeros(self.ACTION_SIZE)
		for i in range(self.ACTION_SIZE):
			Q_arr[i] = self.Q[zero_index + i*self.ACTION_SPACE_SIZE]
		if action:
			return np.argmax(Q_arr)
		else:
			return np.amax(Q_arr)

	def get_next_action(self, state, epsilon):
		if (np.random.uniform() < epsilon):
			return np.random.randint(0, self.ACTION_SIZE)
		else:
			return self.read_q(state, 1)

	# Computes ||Q_{i+1} - Q_i||_2
	def compute_qnorm(self, Q_old):
		return np.linalg.norm(np.subtract(Q_old,self.Q), 2)

	def begin_training(self, max_episodes):
		env = gym.make(self.game)
		self.ACTION_SIZE = env.action_space.n
		self.ACTION_SPACE_SIZE = vec_product(self.buckets)
		self.STATE_SIZE = np.shape(env.observation_space)[0]
		self.Q = np.zeros(self.ACTION_SIZE*self.ACTION_SPACE_SIZE)
		self.CUM_PROD = get_cum_prod(self.buckets)
		epsilon_decay_rate = (self.epsilon_min - self.epsilon_max)/max_episodes
		q_norm_plot_array = np.zeros(max_episodes)
		q_avg_plot_array = np.zeros(max_episodes)
		reward_plot_array = np.zeros(max_episodes)
		for episode in range(max_episodes):
			done = False
			current_state = env.reset()
			epsilon = self.epsilon_max + epsilon_decay_rate*episode
			total_reward = 0
			Q_old = copy.copy(self.Q)
			while not done:
				action = self.get_next_action(current_state, epsilon)
				new_state, reward, done, _ = env.step(action)
				index = self.get_q_index(action, current_state)
				self.Q[index] += self.alpha*(reward + self.gamma*self.read_q(new_state, 0) - self.Q[index])
				current_state = new_state
				if (episode > (max_episodes - 10)):
					env.render()
				total_reward += reward
			reward_plot_array[episode] = total_reward
			q_norm_plot_array[episode] = self.compute_qnorm(Q_old)
			q_avg_plot_array[episode] = np.mean(self.Q)
		figure = plt.figure()
		ax1 = figure.add_subplot(1, 3, 1)
		ax2 = figure.add_subplot(1, 3, 3)
		ax1.plot(q_norm_plot_array)
		ax2.plot(q_avg_plot_array)
		ax1.set_xlabel('Episode')
		ax1.set_ylabel('Norm')
		ax2.set_xlabel('Episode')
		ax2.set_ylabel('Q average')
		plt.show()