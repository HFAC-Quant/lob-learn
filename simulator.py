
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
from tensorflow import keras

num_actions = 11 #5
num_times = 5
num_volumes = 5
total_volume = 2000 #10000 # make entrypoint later to get these values
time_horizon = 50 #1000
is_buy = True
is_dp = True
actions = []
Q_EPSILON = 0.000001    # defines convergence epsilon for qvals and DP
DP_EPSILON = 0.003

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

def get_data(is_buy, start_from_beginning, time_horizon, num_slices, path=None):
	for file in generate_data(read, path, is_buy, is_dp=True, start_from_beginning=start_from_beginning,
							  slice_size=time_horizon, num_slices=num_slices):
		if is_dp:
			for slice in file:
				yield slice.iloc[:, 0:11], slice.iloc[:, 11:21], slice.iloc[:, 21:22] #prices,volumes,midpoint
		else:
			for slice in file:
				yield (slice.iloc[:, 0:5], slice.iloc[:, 5:10], slice.iloc[:, 10:11])

def simulate(is_buy, start_from_beginning, num_episodes, path=None, optpolicy=None):
	rewards = 0
	volume_left = total_volume
	data_gen = get_data(is_buy, start_from_beginning, time_horizon, 1, path)
	prices, volumes, midpoint = next(data_gen)  # get market data from the current time
	theoretical_price = midpoint.iloc[0]

	for num_time in range(num_times):
		#action = algo(prices, volumes, midpoint, volume_left, qvals, optpolicy, num_time)
		action = simulate_action(prices, volumes, midpoint, volume_left, optpolicy, num_time)

		action_price = prices.iloc[int(num_time * time_horizon / num_times), action]
		cur_reward = 0

		for t in range(int(time_horizon / num_times)):
			cur_rew, volume_left = reward(action_price, prices.iloc[int(num_time * time_horizon / num_times) + t],
										  volumes.iloc[int(num_time * time_horizon / num_times) + t], theoretical_price,
										  volume_left)
			cur_reward += cur_rew
			# print(cur_reward)

		rewards += cur_reward
	# print(f"simulation reward: {rewards.item()}, volume_left: {volume_left}")
	return rewards.item(), volume_left


def simulate_set_and_leave(is_buy, start_from_beginning, critical_time_left=int(0.2*time_horizon), path=None, price=None):
	data_gen = get_data(is_buy, start_from_beginning, time_horizon, 1, path)
	prices, volumes, midpoint = next(data_gen)
	theoretical_price = midpoint.iloc[0]
	rewards = 0

	if price is None:
		price = theoretical_price

	volume_left = total_volume

	for t in range(time_horizon - critical_time_left):

		cur_rew, volume_left = reward(price.iloc[0], prices.iloc[t], volumes.iloc[t], theoretical_price, volume_left)
		rewards += cur_rew

	for t in range(critical_time_left):
		if volume_left <= 0:
			break

		cur_rew, volume_left = reward((prices.iloc[t, 10] if is_buy else prices.iloc[t, 0]), prices.iloc[time_horizon - critical_time_left],
									  volumes.iloc[time_horizon - critical_time_left], theoretical_price, volume_left)

		rewards += cur_rew

	# print(f"simulation reward: {rewards.item()}, volume_left: {volume_left}")
	return rewards.item(), volume_left


# def qsimulate(algo, is_buy, qvals=None, optpolicy=None):
# 	rewards = 0
# 	volume_left = total_volume
# 	data_gen = get_data(is_buy, time_horizon)
# 	for time_left in range(time_horizon-1, -1, -1):
# 		prices, volumes, midpoint = next(data_gen) # get market data from the current time
# 		ix = time_horizon - time_left - 1
# 		action = algo(prices, volumes, midpoint, time_left, volume_left, qvals, optpolicy, max(math.ceil(ix/time_horizon * num_times) - 1, 0))
# 		cur_reward, volume_left = reward(action, prices.iloc[ix], volumes.iloc[ix], midpoint.iloc[ix], volume_left)
# 		rewards += cur_reward
# 	print(f"simulation reward: {rewards}, volume_left: {volume_left}")
# 	return rewards, volume_left

def reward(action_price, prices, volumes, midpoint, volume_left): #assumes is_buy is true; fix action vs action price in qlearning
	'''
	action is 0-4, where 0 is buy/sell at lowest level, 4 is buy/sell at highest level
	returns trading costs, i.e. total differences between buy (sell) price and midpoint
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

def dp(is_buy, start_from_beginning, num_episodes, path=None):
	data_gen = get_data(is_buy, start_from_beginning, time_horizon, num_episodes, path)
	dptable = np.zeros((num_times + 1,num_volumes,num_actions)) #num_times + 1 for the penalty if you run out of time and there's still vol and OOB
	volume_left = total_volume
	time_left = time_horizon
	ntable = np.zeros((num_times,num_volumes,num_actions))
	for volume_bucket in range(1,num_volumes): # what if vol_left is > 0 but in the first vol bucket?
		for action in range(num_actions):
			dptable[num_times][volume_bucket][action] = -2000 * volume_bucket #potentially more precise if this is reward of trading volume_left at worst prices at time_horizon
	#print(prices.iloc[0:5], volumes.iloc[0:5], midpoint[0:5])


	dptable_last = dptable.copy()
	for i in range(num_episodes):
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

				# print(f"run {i} dptable:", dptable)

		if i % 10 == 0:
			print(f"Training run {i}")
		# print(f"Relative norm: {np.linalg.norm(dptable - dptable_last)/np.linalg.norm(dptable)}")

		if np.linalg.norm(dptable - dptable_last)/np.linalg.norm(dptable) < DP_EPSILON and i > 0.1 * num_episodes:
			print(f"Breaking at iteration {i}")
			break

		dptable_last = dptable.copy()

	# optpolicy = -np.ones((num_times,num_volumes),dtype=int)
	# for t in range(num_times):
	# 	for v in range(num_volumes):
	# 		optpolicy[t][v] = np.argmax(dptable[t][v]) #optimal action: 0 - 10

	# optpolicy = np.argmax(dptable[:-1], axis=2)
	
	# print("optpolicy:", optpolicy)

	np.save("dptable", dptable)
	return dptable

def simulate_action(prices, volumes, midpoint, volume_left, optpolicy, num_time):
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
	#actions.append(action)
	return action


# def execute(action, prices, volumes, midpoint, time_left, volume_left):
# 	# compute the new state
# 	# time_left -= 1
# 	# t = max(math.ceil(time_left / time_horizon * num_times) - 1, 0)



# 	# compute the reward and volume left
# 	r, volume_left = reward(action, prices, volumes, midpoint, volume_left)
# 	v = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
# 	return r, (t, v), time_left, volume_left


def qlearning_test(prices, volumes, midpoint, volume_left, qvals, optpolicy, num_time):
	#t = max(math.ceil(time_left / time_horizon * num_times) - 1, 0)
	t = num_time
	v = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
	return np.argmax(qvals[t][v])


# Split into another file       
def qlearning(prices, volumes, midpoint, discountrate, learningrate, qvals):
	'''
	qlearning for training
	'''
	s = (0, num_volumes - 1) # start at highest time horizon and volume
	num_time = 0
	volume_bucket = num_volumes - 1
	theoretical_price = midpoint.iloc[0]
	volume_left = total_volume

	# qvals[0][4][0] -= 1000
	# qvals[0][4][1] -= 1000
	# qvals[0][4][2] -= 1000
	# qvals[0][4][3] -= 1000
	# qvals[0][3][0] -= 1000
	# qvals[0][3][1] -= 1000
	# qvals[0][3][2] -= 1000
	# qvals[0][3][3] -= 1000

	while num_time < num_times and volume_left > 0:
		#i = time_horizon - time_left
		a = select_action(s, qvals)  # choose an action according to some algorithm, like epsilon-greedy
		#global actions
		actions.append(a)
		action_price = prices.iloc[int(num_time * time_horizon / num_times), a]
		cur_reward = 0
		# get a reward and go to state s_prime after you execute action a
		for t in range(int(time_horizon/num_times)):
			cur_rew, volume_left = reward(action_price, prices.iloc[int(num_time * time_horizon / num_times) + t], 
				volumes.iloc[int(num_time * time_horizon / num_times) + t], theoretical_price, volume_left)
			cur_reward += cur_rew
		volume_bucket = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
		num_time += 1
		s_prime = (num_time, volume_bucket)
		r = cur_reward

		# r, s_prime, time_left, volume_left = execute(a, prices.iloc[i], volumes.iloc[i], midpoint.iloc[i],
		# 											 time_left, volume_left)
		highest_qval = np.amax(qvals[s_prime[0]][s_prime[1]])
		qvals[s[0]][s[1]][a] = (1-learningrate)*qvals[s[0]][s[1]][a]+learningrate*(r+discountrate*highest_qval)
		s = s_prime
		
	# if volume_left > 0:
	# 	qvals[s[0]][s[1]] -= 1000

	return qvals

def to_optpolicy(valuetable):
	optpolicy = np.argmax(valuetable[:-1], axis=2)
	return optpolicy

def qlearning_training(is_buy, start_from_beginning, discountrate, learningrate, num_episodes=1000, path=None):
	qvals = np.zeros(shape=(num_times + 1, num_volumes, num_actions))
	for volume_bucket in range(1,num_volumes): # what if vol_left is > 0 but in the first vol bucket?
		for action in range(num_actions):
			qvals[num_times][volume_bucket][action] = -2000 * volume_bucket 

	data_gen = get_data(is_buy, start_from_beginning, time_horizon, num_episodes, path)

	qvals_last = qvals.copy()
	for i in range(num_episodes): # train on contiguous blocks of data
		#global actions
		
		prices, volumes, midpoint = next(data_gen) # get market data from the current time
		qvals = qlearning(prices, volumes, midpoint, discountrate, learningrate, qvals)

		if i % 50 == 0:
			print(f"Training run {i}")
		# print(f"Relative norm: {np.linalg.norm(qvals - qvals_last)/np.linalg.norm(qvals)}")

		if np.linalg.norm(qvals - qvals_last)/np.linalg.norm(qvals) < Q_EPSILON and i > 0.1 * num_episodes:
			print(f"Breaking at iteration {i}")
			break

		qvals_last = qvals.copy()
	
	#global actions

	actions_np = np.asarray(actions)
	# n, bins, patches = plt.hist(actions_np, facecolor='blue', alpha=0.5)
	# plt.show()
	np.save("qvals", qvals)
	#print("actions:",actions)
	return qvals

def train_simulate():
	SAMPLE_PATH_LST = ['data/order_books/2011/IF1101.csv','data/order_books/2012/IF1201.csv',
					   'data/order_books/2013/IF1301.csv','data/order_books/2014/IF1401.csv']
	strings = ['trainpath1101', 'trainpath1201', 'trainpath1301', 'trainpath1401']

	dflist = []
	for counter,trainpath in enumerate(SAMPLE_PATH_LST):
		print(f"Training Q-learning on {trainpath}...")
		qlearning_training(True, False, 0.1, 0.6, num_episodes=2000, path=trainpath)
		qvals = np.load("qvals.npy")

		print(f"Training DP on {trainpath}...")
		dp(True, False, 60, path=trainpath)
		dptable = np.load("dptable.npy")

		print(f"Training Keras on {trainpath}...")
		keras_learning(True, False, 600, path=trainpath)
		kmodel = keras.models.load_model('keras_model')

		for simulpath in SAMPLE_PATH_LST:
			print(f"Simulating on {simulpath}")
			qrewards, qvolume_left = simulate(True, False, 1, path=simulpath, optpolicy=to_optpolicy(qvals))
			dprewards, dpvolume_left = simulate(True, False, 1, path=simulpath, optpolicy=to_optpolicy(dptable))
			slrewards, slvolume_left = simulate_set_and_leave(True, False, path=simulpath)
			krewards, kvolume_left = keras_testing(True, False, path=simulpath, model=kmodel)

			dftemp = pd.DataFrame({'Training Path':trainpath, 'Simulation Path':simulpath,
				'Q-Value Rewards':qrewards,'Q-Value Volume Left':qvolume_left,
				'DP Rewards':dprewards,'DP Volume Left':dpvolume_left, 'Keras Rewards':krewards,
				'Keras Volume Left':kvolume_left, 'Set and Leave Rewards':slrewards,
				'Set and Leave Volume Left':slvolume_left}, index=[0])
			dflist.append(dftemp)

		df = pd.concat(dflist).sort_values(['Training Path','Simulation Path'],ascending=True)
		df.to_csv('trainsimulate1128_' + strings[counter] + '.csv')

	return df


# def train_simulate_keras():
# 	df = pd.read_csv("trainsimulate0726_v2_trainpath1401.csv")
# 	SAMPLE_PATH_LST = ['data/order_books/2011/IF1101.csv', 'data/order_books/2012/IF1201.csv',
# 					   'data/order_books/2013/IF1301.csv','data/order_books/2014/IF1401.csv']
#
# 	reward_list = []
# 	volume_left_list = []
#
# 	for trainpath in SAMPLE_PATH_LST:
# 		keras_learning(1000, trainpath, True, True, time_horizon)
# 		for simulpath in SAMPLE_PATH_LST:
# 			reward, volume_left = keras_testing(True, True, simulpath)
# 			reward_list.append(reward)
# 			volume_left_list.append(volume_left)
#
# 	df['Keras Rewards'] = reward_list
# 	df['Keras Volume Left'] = volume_left_list
#
# 	df.to_csv('trainsimulate1128.csv')
#
# 	return df


def to_state(time_bucket, volume_bucket):
	# For keras learning
	return num_times * volume_bucket + time_bucket


def to_buckets(state):
	# For keras learning
	volume_bucket = int(state / num_times)
	time_bucket = state % num_times
	return (time_bucket, volume_bucket)


def keras_learning(is_buy, start_from_beginning, num_episodes, gamma=0.95, eps=0.2, decay_factor=0.99, path=None):
	dim = num_times * num_volumes
	model = keras.Sequential()
	model.add(keras.layers.InputLayer(batch_input_shape=(1,dim)))
	model.add(keras.layers.Dense(25, activation='sigmoid'))
	model.add(keras.layers.Dense(num_actions, activation='linear'))
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])

	data_gen = get_data(is_buy, start_from_beginning, time_horizon, num_episodes, path)

	r_list = []
	v_list = []
	for i in range(num_episodes):
		prices, volumes, midpoint = next(data_gen)

		s = to_state(0, num_volumes - 1)  # start at time bucket 0 and highest volume bucket
		eps *= decay_factor
		if i % 50 == 0:
			print("Episode {} of {}".format(i+1, num_episodes))
		r_sum = 0

		num_time = 0   # this is a bucket
		volume_left = total_volume
		theoretical_price = midpoint.iloc[0]

		# Select an action
		while num_time < num_times and volume_left > 0:
			if np.random.random() < eps:
				a = np.random.randint(0, num_actions)
			else:
				a = np.argmax(model.predict(np.identity(dim)[s:s + 1]))

			# print(f"action: {a}")

			action_price = prices.iloc[int(num_time * time_horizon / num_times), a]

			# calculate reward and new state
			cur_reward = 0
			for t in range(int(time_horizon / num_times)):
				cur_rew, volume_left = reward(action_price, prices.iloc[int(num_time * time_horizon / num_times) + t],
											  volumes.iloc[int(num_time * time_horizon / num_times) + t],
											  theoretical_price, volume_left)
				cur_reward += cur_rew.item()
			r = cur_reward
			new_vol_bucket = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
			num_time += 1
			s_prime = to_state(num_time, new_vol_bucket)

			# print(f"dim = {dim}, num_time = {num_time}, new_vol_bucket = {new_vol_bucket}, s_prime = {s_prime}")

			if num_time < num_times:
				target = r + gamma * np.max(model.predict(np.identity(dim)[s_prime:s_prime + 1]))
			else:
				target = -2000 * new_vol_bucket
			target_vec = model.predict(np.identity(dim)[s:s + 1])[0]
			target_vec[a] = target
			model.fit(np.identity(dim)[s:s + 1], target_vec.reshape(-1, num_actions), epochs=1, verbose=0)
			s = s_prime
			r_sum += r

		r_list.append(r_sum)
		v_list.append(volume_left)

	model.save('keras_model')
	# print(f"Rewards:\n{r_list}")
	# print(f"Volume left:\n{v_list}")

	# to load:
	# model = keras.models.load_model('keras_model')
	# state = to_state(0, num_volumes-1)
	# model.predict(np.identity(dim)[state:state + 1])


def keras_testing(is_buy, start_from_beginning, path=None, model=None):
	dim = num_times * num_volumes
	if model is None:
		model = keras.models.load_model('keras_model')

	rewards = 0
	volume_left = total_volume
	data_gen = get_data(is_buy, start_from_beginning, time_horizon, 1, path)
	prices, volumes, midpoint = next(data_gen)  # get market data from the current time
	theoretical_price = midpoint.iloc[0]

	s = to_state(0, num_volumes - 1)  # start at time bucket 0 and highest volume bucket

	for num_time in range(num_times):
		action = np.argmax(model.predict(np.identity(dim)[s:s + 1]))

		action_price = prices.iloc[int(num_time * time_horizon / num_times), action]
		cur_reward = 0

		for t in range(int(time_horizon / num_times)):
			cur_rew, volume_left = reward(action_price, prices.iloc[int(num_time * time_horizon / num_times) + t],
										  volumes.iloc[int(num_time * time_horizon / num_times) + t], theoretical_price,
										  volume_left)
			cur_reward += cur_rew
			print(cur_reward)
		rewards += cur_reward

		new_vol_bucket = max(math.ceil(volume_left / total_volume * num_volumes) - 1, 0)
		num_time += 1
		s = to_state(num_time, new_vol_bucket)

	# print(f"simulation reward: {rewards.item()}, volume_left: {volume_left}")
	return rewards.item(), volume_left

#df = train_simulate()
#df.to_csv('trainsimulate0726_total.csv')


#training(discountrate=0.1, learningrate=0.6, num_iter=500)

# # print(qvals)
# # print(qvals_to_optpolicy(qvals))
#

#qvals = np.load("qvals.npy")

# # print("qvals:")

#simulate(is_buy=True, optpolicy=to_optpolicy(qvals))

#simulation reward: 1.0872285918271558, volume_left: 0 (data 1201)
#simulation reward: 0.17933092898893416, volume_left: 1842 (with qvals from training on 1201; data 1102)
#simulation reward: 1.6027771843623007, volume_left: 323 (trained on 1201; 1401)
#simulation reward: 1.0793420209356894, volume_left: 288 (trained on 1401; 1401)
#simulation reward: 2.0309335612048565, volume_left: 0 (trained on 1401 with 247 iter; 1401 simulate)
#simulation reward: 0.0024436600597323955, volume_left: 1953 (trained on 1307 with 442 iter; 1307 simulate)


#dp(is_buy=True, num_iter=50)
# dptable = np.load("dptable.npy")

# # print("dptable:")

# simulate(is_buy=True, optpolicy=to_optpolicy(dptable))

#simulation reward: 1.0007139557346063, volume_left: 0 (data 1201)
#simulation reward: 0.15327695560254007, volume_left: 1842 (with dptable from training on 1201; data 1102)
#simulation reward: 2.302414014099577, volume_left: 0 (trained on 1201; 1401)
#simulation reward: 2.266438795129272, volume_left: 0 (trained on 1401; 1401)
#simulation reward: 1.2599871822260045, volume_left: 0 (trained on 1401 with 100 iter; 1401 simulate)
#simulation reward: 2.266438795129272, volume_left: 0 (trained on 1401 with 3 iter; 1401 simulate)
#simulation reward: 0.037391877739417365, volume_left: 1922 (trained on 1307 with 79 iter; 1307 simulate)



#simulate_set_and_leave(is_buy=True)

#simulation reward: 1.4502029480880052, volume_left: 0 (1401 simulate)
#simulation reward: 0.07082735347735696, volume_left: 1774 (1307 simulate)


# print(qsimulate(qlearning_test, is_buy, qvals))


# Test keras model:

# keras_learning(1000, "data/order_books/2012/IF1201.csv", True, True, time_horizon)
# keras_testing(True, True, "data/order_books/2014/IF1104.csv")
# simulation reward: 1.0537146696903235, volume_left: 0 (data 1201)

# train_simulate_keras()


# Final comparison
train_simulate()
