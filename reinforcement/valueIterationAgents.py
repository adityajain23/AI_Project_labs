#==============================
# Group 8
# Team Members:
# Aditya Jain : 1903102
# Adwait Agashe: 1903103
# Gunjan Mayekar: 1903117
#==============================


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Discount factor
        Y = self.discount

        for _ in range(self.iterations):
            # Created temp dict for value function "Vnew"
            Vnew = util.Counter()
            # run for each state
            for state in self.mdp.getStates():
                # if state is not terminal
                if not self.mdp.isTerminal(state):
                    # update the value of state as max(Q(s,a) for all actions a)
                    action = self.getAction(state)
                    Vnew[state] = self.getQValue(state, action)
            # update the old value function to new value function
            self.values = Vnew

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Discount factor
        Y = self.discount
        curr = 0
        # for all possible transitions from the state 's' after taking action 'a'
        for k in self.mdp.getTransitionStatesAndProbs(state, action):
            nextState = k[0]
            prop = k[1]
            reward = self.mdp.getReward(state, action, nextState)
            # update the curr value based on the formula
            curr = curr + prop * (reward + Y * self.values[nextState])
        return curr

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        # if the state is terminal then return None
        if self.mdp.isTerminal(state):
            return None

        # Discount factor
        Y = self.discount
        temp = float("-inf")
        actions = self.mdp.getPossibleActions(state)
        action = actions[0]
        # for each legal action get the Qvalue for the (state, action) pair and
        # update the best action if Qvalue of the curr_action is better than the
        # previous best.
        for j in actions:
            curr = self.computeQValueFromValues(state, j)
            if curr > temp:
                temp = curr
                action = j
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Discount factor
        Y = self.discount
        itr = self.iterations
        # for each iteration update the value of one state per iteration as
        # given in the algorithm
        while itr > 0:
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    action = self.getAction(state)
                    self.values[state] = self.getQValue(state, action)
                # reducing iteration after every state value update
                itr = itr - 1
                if itr == 0:
                    break


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # created a function which returns the predecessors of all states
    def predecessors(self):
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(
                        state, action
                    ):
                        # if the nextState is in predecessors then we add the curr_state
                        # otherwise we inilize a new dict for the nextState
                        if nextState in predecessors:
                            predecessors[nextState].add(state)
                        else:
                            predecessors[nextState] = {state}
        return predecessors

    def runValueIteration(self):
        # inilizing priority queue
        pq = util.PriorityQueue()
        # getting predecessors for every state
        predecessors = self.predecessors()

        # for each non-terminal state add the state to the priority queue
        # according to the condition given in the algorithm
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                action = self.getAction(state)
                Qvalue = self.getQValue(state, action)
                diff = abs(Qvalue - self.values[state])
                pq.update(state, -diff)

        # for each iteration
        for i in range(self.iterations):
            # if queue is empty then exit the loop
            if pq.isEmpty():
                break
            # get the element with highest priority from the queue
            temp_state = pq.pop()
            # if the temp_state is not in terminal then update its value
            if not self.mdp.isTerminal(temp_state):
                action = self.getAction(temp_state)
                Qvalue = self.getQValue(temp_state, action)
                self.values[temp_state] = Qvalue
            # for all the predecessors update them into the queue if they are 
            # non-terminal according to the condition given in the algorithm
            for p in predecessors[temp_state]:
                if not self.mdp.isTerminal(p):
                    action = self.getAction(p)
                    Qvalue = self.getQValue(p, action)
                    diff = abs(Qvalue - self.values[p])
                    if diff > self.theta:
                        pq.update(p, -diff)
