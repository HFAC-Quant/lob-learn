
# Exchange: go down the list of bids and asks, algorithm decides whether to buy, sell, or do nothing

# Exchange module: take in states and actions 

#use yield to send over time horizon, amount to trade in data

# midpoint_price <- 

#independent contracts, train with each of them, simulate with one contract at a time

import numpy

def get_data(cleandata):
	for line in lines:
		get info and yield

def qlearning(states, actions, discountrate, learningrate):
	qvals = {}
	for state in states:
		for action in actions:
			qvals[(state,action)] = 0
	s = numpy.random.choice(states)
	while (training_period):
		a = select_action(s,qvals) #choose an action according to some algorithm, like epsilon-greedy
		r,s_prime = execute(a) #get a reward and go to state s_prime after you execute action a
		action_q = []
		for (s,a) in qvals:			
			if s_prime==s:
				action_q.append((a,qvals[(s,a)]))
		(best_action, highest_qval) = max(action_q, key=lambda x:x[1])
		qvals[(s,a)] = (1-learningrate)*qvals[(s,a)]+learningrate*(r+discountrate*highest_qval)
		s = s_prime
	return (best_action,qvals)

def simulate(qlearning):
	rewards = 0
	for time in time_horizon:	
		states, actions = get_data(cleandata) #get market data from the current time
		best_action, qvals = qlearning(states, actions, discountrate=0.1, learningrate=0.1) #pick best action from qvals corresponding to current state, execute
		rewards += qvals[(curr_state, best_action)]
	# bucketize states (volume left to sell) logarithmically
