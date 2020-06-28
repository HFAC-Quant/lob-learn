
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

num_actions = 11 #5
num_times = 5
num_volumes = 5
total_volume = 2000 #10000 # make entrypoint later to get these values
time_horizon = 50 #1000
is_buy = True
is_dp = True
actions = []

# TODO June 28: Rewrite q-learning to match DP
# TODO June 28: Compare DP against q-learning against set and leave
# TODO June 28: Use Keras/diff model

# period = 30 #30 seconds
# timedelta = .5 #time btw prices in order book


#TEST RESULTS: 2012/IF1201
# num_times = 5
# num_volumes = 5
# total_volume = 1000 
# time_horizon = 50 

# optpolicy: [[ 0  0  0  0  0]
#  [ 7 10 10 10 10]
#  [ 7  7  7  7 10]
#  [ 0  0  1  1  4]
#  [ 8  9  9  9  9]]

# num_times = 5
# num_volumes = 5
# total_volume = 2000 
# time_horizon = 50 
# optpolicy: [[ 0  0  0  2  1]
#  [10 10 10 10 10]
#  [ 7  7 10 10 10]
#  [ 0  3  4 10 10]
#  [ 9  9 10 10 10]]

# same optpolicy for 2012/IF1209, 2013/IF1301 (same parameters as above)

# IF 1005: 
# [[ 5  5  5  5  5]  
# [10 10 10 10 10]  
# [ 6  6  6  6  6]  
# [ 0  0  0  0  0]  
# [ 0  0  0  0  0]]

def get_data(is_buy, time_horizon):
	for file in generate_data(read, is_buy, is_dp=True, slice_size=time_horizon):
		if is_dp:
			for slice in file:
				yield slice.iloc[:, 0:11], slice.iloc[:, 11:21], slice.iloc[:, 21:22] #prices,volumes,midpoint
		else:
			for slice in file:
				yield (slice.iloc[:, 0:5], slice.iloc[:, 5:10], slice.iloc[:, 10:11])

def simulate(algo, is_buy, qvals=None, optpolicy=None):
	rewards = 0
	volume_left = total_volume
	data_gen = get_data(is_buy, time_horizon)
	prices, volumes, midpoint = next(data_gen)  # get market data from the current time
	theoretical_price = midpoint.iloc[0]
	for num_time in range(num_times):
		time_left = (num_times - num_time) * time_horizon / num_times
		action = algo(prices, volumes, midpoint, time_left, volume_left, qvals, optpolicy, num_time)
		action_price = prices.iloc[int(num_time * time_horizon / num_times), action]
		cur_reward = 0

		for t in range(int(time_horizon / num_times)):
			cur_rew, volume_left = reward(action_price, prices.iloc[int(num_time * time_horizon / num_times) + t],
										  volumes.iloc[int(num_time * time_horizon / num_times) + t], theoretical_price,
										  volume_left)
			print(cur_rew)
			print(volume_left)
			cur_reward += cur_rew

		rewards += cur_reward
	print(f"simulation reward: {rewards.item()}, volume_left: {volume_left}")
	return rewards, volume_left

def qsimulate(algo, is_buy, qvals=None, optpolicy=None):
	rewards = 0
	volume_left = total_volume
	data_gen = get_data(is_buy, time_horizon)
	for time_left in range(time_horizon-1, -1, -1):
		prices, volumes, midpoint = next(data_gen) # get market data from the current time
		ix = time_horizon - time_left - 1
		action = algo(prices, volumes, midpoint, time_left, volume_left, qvals, optpolicy, max(math.ceil(ix/time_horizon * num_times) - 1, 0))
		cur_reward, volume_left = reward(action, prices.iloc[ix], volumes.iloc[ix], midpoint.iloc[ix], volume_left)
		rewards += cur_reward
	print(f"simulation reward: {rewards}, volume_left: {volume_left}")
	return rewards, volume_left

def reward(action_price, prices, volumes, midpoint, volume_left): #assumes is_buy is true; fix action vs action price in qlearning
	'''
	action is 0-4, where 0 is buy/sell at lowest level, 4 is buy/sell at highest level
	'''
	cur_reward = 0
	if is_buy:
		newprices, newvolumes = prices.iloc[6:11], volumes.iloc[5:10]
		for i in range(5):
			if action_price >= newprices.iloc[i]: #trade
				cur_reward -= (newprices.iloc[i] - midpoint)*min(newvolumes.iloc[i], volume_left) #assumes is_buy is true
				volume_left -= min(newvolumes.iloc[i], volume_left)
				if volume_left <= 0:
					break
	else:
		newprices, newvolumes = prices.iloc[4:-1:-1], volumes.iloc[4:-1:-1]
		for i in range(5):
			if action_price <= newprices.iloc[i]:
				cur_reward -= (midpoint - newprices.iloc[i])*min(newvolumes.iloc[i], volume_left)
				volume_left -= min(newvolumes.iloc[i], volume_left)
				if volume_left <= 0:
					break
	return cur_reward/midpoint, volume_left

def dp(is_buy, learningrate, num_iter=1):
	data_gen = get_data(is_buy, time_horizon)
	dptable = np.zeros((num_times + 1,num_volumes,num_actions)) #num_times + 1 for the penalty if you run out of time and there's still vol and OOB
	volume_left = total_volume
	time_left = time_horizon
	ntable = np.zeros((num_times,num_volumes,num_actions))
	for volume_bucket in range(1,num_volumes): # what if vol_left is > 0 but in the first vol bucket?
		for action in range(num_actions):
			dptable[num_times][volume_bucket][action] = -2000 * volume_bucket #potentially more precise if this is reward of trading volume_left at worst prices at time_horizon
	#print(prices.iloc[0:5], volumes.iloc[0:5], midpoint[0:5])

	for i in range(num_iter):
		prices, volumes, midpoint = next(data_gen)
		theoretical_price = midpoint.iloc[0]
		for num_time in range(num_times - 1,-1,-1): #num_time is time that has elapsed (in buckets)
			for volume_bucket in range(num_volumes):
				for action in range(num_actions):

					# if num_time == 1 and volume_bucket == 0:
					# 	print("HERE")

					state = (num_time, volume_bucket)

					cur_reward = 0
					volume_left = (volume_bucket + 1) * total_volume / num_volumes #top of volume bucket

					# if num_time == 1 and volume_bucket == 0:
					# 	print(f"Volume_left = {volume_left}")

					action_price = prices.iloc[int(num_time * time_horizon / num_times), action]

					for t in range(int(time_horizon/num_times)):
						cur_rew, volume_left = reward(action_price, prices.iloc[int(num_time * time_horizon / num_times) + t], volumes.iloc[int(num_time * time_horizon / num_times) + t], theoretical_price, volume_left)
						cur_reward += cur_rew
						#print(f"action_price = {action_price}")
					new_vol_bucket = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)

					# if num_time == num_times - 1:
					# 	print(f"volume left = {volume_left}")

					# if num_time == 1 and volume_bucket == 0:
					# 	print(f"new_vol_bucket = {new_vol_bucket}")

					best_reward = np.amax(dptable[num_time + 1][new_vol_bucket])

					# if num_time == 1 and volume_bucket == 0:
					# 	print(f"best_reward = {best_reward}")

					ntable[num_time][volume_bucket][action] += 1
					n = float(ntable[num_time][volume_bucket][action])

					dptable[num_time][volume_bucket][action] = n/(n+1) * dptable[num_time][volume_bucket][action] + 1/(n+1) * (cur_reward + best_reward)

					# if num_time == 1 and volume_bucket == 0:
					# 	print(f"Updated to {dptable[num_time][volume_bucket][action]}")

				print(f"run {i} dptable:", dptable)

	optpolicy = -np.ones((num_times,num_volumes),dtype=int)
	for t in range(num_times):
		for v in range(num_volumes):
			optpolicy[t][v] = np.argmax(dptable[t][v]) #optimal action: 0 - 10
	
	print("optpolicy:", optpolicy)
	return dptable, optpolicy

def dp_action(prices, volumes, midpoint, time_left, volume_left, qvals, optpolicy, num_time):
	volume_bucket = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
	return optpolicy[num_time][volume_bucket]


# def reward2(midpoint,cur_cost):
# 	return -abs(cur_cost - midpoint)/midpoint * 10**4 #basis points


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
		all_qvals = qvals[state[0]][state[1]]
		action = np.random.choice(np.flatnonzero(all_qvals == all_qvals.max()))
	actions.append(action)
	return action


def execute(action, prices, volumes, midpoint, time_left, volume_left):
	# compute the new state
	time_left -= 1
	t = max(math.ceil(time_left / time_horizon * num_times) - 1, 0)

	# compute the reward and volume left
	r, volume_left = reward(action, prices, volumes, midpoint, volume_left)
	v = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
	return r, (t, v), time_left, volume_left


def qlearning_test(prices, volumes, midpoint, time_left, volume_left, qvals, optpolicy, num_time):
	t = max(math.ceil(time_left / time_horizon * num_times) - 1, 0)
	v = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
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
		global actions
		actions = []
		print(f"Training run {i}")
		prices, volumes, midpoint = next(data_gen) # get market data from the current time
		qvals = qlearning(prices, volumes, midpoint, discountrate, learningrate, qvals)
		print(qvals)
	actions_np = np.asarray(actions)
	n, bins, patches = plt.hist(actions_np, facecolor='blue', alpha=0.5)
	plt.show()
	np.save("qvals", qvals)
	return qvals

#training(discountrate=0.1, learningrate=0.6)

# dptable, optpolicy = dp(is_buy=True, learningrate=0.6)
# simulate(dp_action, is_buy=True, optpolicy=optpolicy)

qvals = np.load("qvals.npy")
print(qsimulate(qlearning_test, is_buy, qvals))

