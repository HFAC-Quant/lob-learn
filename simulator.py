
# Exchange: go down the list of bids and asks, algorithm decides whether to buy, sell, or do nothing

# Exchange module: take in states and actions 

#use yield to send over time horizon, amount to trade in data

# midpoint_price <- 

#independent contracts, train with each of them, simulate with one contract at a time

import numpy
import random
from data_loaders import *

num_actions = 6
num_times = 5
num_volumes = 5
total_volume = 100 # make entrypoint later to get these values
time_horizon = 100

def get_data(is_buy, time_horizon):
	for file in generate_data(read, is_buy, slice_size=time_horizon):
		for slice in file:
			yield (slice[:,0:5], slice[:,5:10], slice[:,10])

def simulate(qlearning, is_buy):
	rewards = 0
	for time in time_horizon:
		prices, volumes, midpoint = get_data(is_buy, time_horizon) # get market data from the current time
		best_action, qvals = qlearning(prices, volumes, midpoint, discountrate=0.1, learningrate=0.1) #pick best action from qvals corresponding to current state, execute
		rewards += qvals[(curr_state, best_action)]
	# bucketize states (volume left to sell) logarithmically

def reward(action, prices, volumes, midpoint, volume_left):
	'''
	action is 0-5, where 0 is don't trade, 5 is buy/sell at highest level
	'''
	cur_reward = 0
	for i in range(action+1):
		cur_reward -= abs(prices[i] - midpoint)*min(volumes[i],volume_left)
		volume_left -= min(volumes[i],volume_left)
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
	if rand < epsilon:
		action = random.randint(0, num_actions-1) #inclusive
	else:
		action = np.argmax(qvals[state[0]][state[1]]) # numpy
	# print(rand, action)
	return action

def execute(action, prices, volumes, midpoint, time_left, volume_left):
	# compute the new state
	time_left -= 1
	t = floor(time_left / time_horizon * num_times)

	# compute the reward and volume left
	r, volume_left = reward(action, prices, volumes, midpoint, volume_left)
	v = floor(vl / total_volume * num_volumes)
	return r, (t, v), time_left, volume_left

# Split into another file       
def qlearning(prices, volumes, midpoint, states, actions, discountrate, learningrate, qvals):
	'''
	qlearning for training
	'''
	s = (num_times - 1, num_volumes - 1) # start at highest time horizon and volume
	time_left = time_horizon
	volume_left = total_volume

	while time_left > 0:
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
		a = select_action(s,qvals) #choose an action according to some algorithm, like epsilon-greedy
		#get a reward and go to state s_prime after you execute action a
		r, s_prime, time_left, volume_left = execute(a, prices[i], volumes[i], midpoint[i], time_left, volume_left)
		highest_qval = np.argmax(qvals[s_prime[0]][s_prime[1]])
		qvals[s[0]][s[1]][a] = (1-learningrate)*qvals[s[0]][s[1]][a]+learningrate*(r+discountrate*highest_qval)
		s = s_prime
	return qvals

def training():
	qvals = np.zeros(shape=(num_times, num_volumes, num_actions))
	num_runs = 10
	for i in range(num_runs): # train on contiguous blocks of data
		prices, volumes, midpoint = get_data(is_buy, time_horizon) # get market data from the current time
		qvals = qlearning(prices, volumes, midpoint, states, actions, discountrate, learningrate, qvals)