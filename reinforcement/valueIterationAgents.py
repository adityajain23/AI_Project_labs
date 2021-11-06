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


from typing import Counter
import mdp
from util import PriorityQueue
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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            temp = self.values.copy()
            possibleStates = self.mdp.getStates()
            for state in possibleStates:
                actions = self.mdp.getPossibleActions(state)
                actionVals = []
                for action in actions:
                    trans = self.mdp.getTransitionStatesAndProbs(state, action)
                    sum = 0
                    for j in range(len(trans)):
                        newstate = trans[j][0]
                        prob = trans[j][1]
                        sum += prob*(self.mdp.getReward(state, action,
                                                        newstate) + self.discount*temp[newstate])
                    actionVals.append(sum)
                    # print(state, action, actionVals, trans)
                if len(actionVals) > 0:
                    self.values[state] = max(actionVals)

                # print(state, self.values)

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
        # util.raiseNotDefined()
        trans = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for j in range(len(trans)):
            newState = trans[j][0]
            prob = trans[j][1]
            sum += prob*(self.mdp.getReward(state, action,
                         newState) + self.discount*self.values[newState])

        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        ma = float("-inf")
        bestAction = actions[0]
        for action in actions:
            sum = self.computeQValueFromValues(state, action)
            if (bestAction == None and ma == float("-inf")) or sum >= ma:
                bestAction = action
                ma = sum
        return bestAction

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
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            temp = self.values.copy()
            possibleStates = self.mdp.getStates()
            l = len(possibleStates)
            # for state in possibleStates:
            state = possibleStates[i % l]
            actions = self.mdp.getPossibleActions(state)
            actionVals = []
            for action in actions:
                trans = self.mdp.getTransitionStatesAndProbs(state, action)
                sum = 0
                for j in range(len(trans)):
                    newstate = trans[j][0]
                    prob = trans[j][1]
                    sum += prob*(self.mdp.getReward(state, action,
                                                    newstate) + self.discount*temp[newstate])
                actionVals.append(sum)
                # print(state, action, actionVals, trans)
            if len(actionVals) > 0:
                self.values[state] = max(actionVals)


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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()

        predecessors = {}
        allStates = self.mdp.getStates()
        for state in allStates:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    trans = self.mdp.getTransitionStatesAndProbs(state, action)
                    for arrow in trans:
                        newState, prob = arrow
                        if newState not in predecessors:
                            predecessors[newState] = []
                        predecessors[newState].append(state)

        pq = PriorityQueue()
        allStates = self.mdp.getStates()

        for state in allStates:
            if not self.mdp.isTerminal(state):
                maxQ = max([self.getQValue(state, action)
                           for action in self.mdp.getPossibleActions(state)])
                diff = abs(maxQ - self.getValue(state))
                pq.update(state, -diff)

        for iteration in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max([self.computeQValueFromValues(state, action)
                                          for action in self.mdp.getPossibleActions(state)])

            preS = predecessors[state]
            for pre in preS:
                if not self.mdp.isTerminal(pre):
                    maxQ = max([self.computeQValueFromValues(pre, action)
                                for action in self.mdp.getPossibleActions(pre)])
                    diff = abs(maxQ - self.getValue(pre))
                    if diff > self.theta:
                        pq.update(pre, -diff)
