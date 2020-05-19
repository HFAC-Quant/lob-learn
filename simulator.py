
# Exchange: go down the list of bids and asks, algorithm decides whether to buy, sell, or do nothing

# Exchange module: take in states and actions 

#use yield to send over time horizon, amount to trade in data

# midpoint_price <- 

#independent contracts, train with each of them, simulate with one contract at a time

import numpy
import random
import math
import matplotlib.pyplot as plt
from data_loaders import *

num_actions = 5
num_times = 5
num_volumes = 5
total_volume = 10000  # make entrypoint later to get these values
time_horizon = 1000
is_buy = True
actions = []


def get_data(is_buy, time_horizon):
	for file in generate_data(read, is_buy, slice_size=time_horizon):
		for slice in file:
			yield slice.iloc[:, 0:5], slice.iloc[:, 5:10], slice.iloc[:, 10]


def simulate(algo, is_buy, qvals=None):
	rewards = 0
	volume_left = total_volume
	data_gen = get_data(is_buy, time_horizon)
	for time_left in range(time_horizon-1, -1, -1):
		prices, volumes, midpoint = next(data_gen) # get market data from the current time
		ix = time_horizon - time_left - 1
		action = algo(prices, volumes, midpoint, time_left, volume_left, qvals)
		cur_reward, volume_left = reward(action, prices.iloc[ix], volumes.iloc[ix], midpoint.iloc[ix], volume_left)
		rewards += cur_reward
	return rewards, volume_left


def reward(action, prices, volumes, midpoint, volume_left):
	'''
	action is 0-4, where 0 is buy/sell at lowest level, 4 is buy/sell at highest level
	'''
	cur_reward = 0
	for i in range(action+1):
		cur_reward -= abs(prices.iloc[i] - midpoint)*min(volumes.iloc[i], volume_left)
		volume_left -= min(volumes.iloc[i], volume_left)
		if volume_left <= 0:
			break
	return cur_reward, volume_left


'''
use to test select_action:
qvals = np.asarray([[[0, 1, 2],
					[3, 4, 5],
					[6, 7, 8]],
					[[9, 10, 11],
					[12, 5, 0],
					[15, 17, 16]], ##
					[[20, 21, 22],
					[23, 24, 25],
					[26, 27, 28]]])
'''


def select_action(state, qvals):
	epsilon = 0.1
	rand = random.uniform(0, 1)
	# if state[0] == 0:
	# 	action = num_actions-1
	if rand < epsilon:
		action = random.randint(0, num_actions-1) #inclusive
	else:
		action = np.argmax(qvals[state[0]][state[1]]) # numpy
	actions.append(action)
	return action


def execute(action, prices, volumes, midpoint, time_left, volume_left):
	# compute the new state
	time_left -= 1
	t = math.ceil(time_left / time_horizon * num_times) - 1

	# compute the reward and volume left
	r, volume_left = reward(action, prices, volumes, midpoint, volume_left)
	v = math.ceil(volume_left / total_volume * num_volumes) - 1
	return r, (t, v), time_left, volume_left


def qlearning_test(prices, volumes, midpoint, time_left, volume_left, qvals):
	t = math.ceil(time_left / time_horizon * num_times) - 1
	v = math.ceil(volume_left / total_volume * num_volumes) - 1
	return np.argmax(qvals[t][v])


# Split into another file       
def qlearning(prices, volumes, midpoint, discountrate, learningrate, qvals):
	'''
	qlearning for training
	'''
	s = (num_times - 1, num_volumes - 1) # start at highest time horizon and volume
	time_left = time_horizon
	volume_left = total_volume

	qvals[0][4][0] -= 1000
	qvals[0][4][1] -= 1000
	qvals[0][4][2] -= 1000
	qvals[0][4][3] -= 1000
	qvals[0][3][0] -= 1000
	qvals[0][3][1] -= 1000
	qvals[0][3][2] -= 1000
	qvals[0][3][3] -= 1000

	while time_left > 0 and volume_left > 0:
		'''
		action_q = []
		for (s,a) in qvals:			
			if s_prime == s:
				action_q.append((a,qvals[(s,a)]))
		(best_action, highest_qval) = max(action_q, key=lambda x:x[1])
		qvals[(s,a)] = (1-learningrate)*qvals[(s,a)]+learningrate*(r+discountrate*highest_qval)
		s = s_prime
		'''
		i = time_horizon - time_left
		a = select_action(s, qvals)  # choose an action according to some algorithm, like epsilon-greedy
		# get a reward and go to state s_prime after you execute action a
		r, s_prime, time_left, volume_left = execute(a, prices.iloc[i], volumes.iloc[i], midpoint.iloc[i],
													 time_left, volume_left)
		highest_qval = np.amax(qvals[s_prime[0]][s_prime[1]])
		qvals[s[0]][s[1]][a] = (1-learningrate)*qvals[s[0]][s[1]][a]+learningrate*(r+discountrate*highest_qval)
		s = s_prime
	# if volume_left > 0:
	# 	qvals[s[0]][s[1]] -= 1000
	return qvals

def training(discountrate, learningrate):
	qvals = np.zeros(shape=(num_times, num_volumes, num_actions))
	num_runs = 100
	data_gen = get_data(is_buy, time_horizon)
	for i in range(num_runs): # train on contiguous blocks of data
		# global actions
		# actions = []
		print(f"Training run {i}")
		prices, volumes, midpoint = next(data_gen) # get market data from the current time
		qvals = qlearning(prices, volumes, midpoint, discountrate, learningrate, qvals)
		print(qvals)
	# actions_np = np.asarray(actions)
	# n, bins, patches = plt.hist(actions_np, facecolor='blue', alpha=0.5)
	# plt.show()
	np.save("qvals", qvals)
	return qvals

# training(discountrate=0.1, learningrate=0.6)

qvals = np.load("qvals.npy")
print(simulate(qlearning_test, is_buy, qvals))