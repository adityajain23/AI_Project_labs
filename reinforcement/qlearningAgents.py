#==============================
# Group 8
# Team Members:
# Aditya Jain : 1903102
# Adwait Agashe: 1903103
# Gunjan Mayekar: 1903117
#==============================

# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Inilizing Q as dict
        self.Q = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        # returning Q-value for (state, action) pair
        return self.Q[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if len(actions) > 0:
            # for all actions put value as max of Qvalues for all (state, action) pairs
            value = max([self.getQValue(state, action) for action in actions])
        else:
            value = 0
        # returing max_value over all actions
        return value

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        # all possible actions
        actions = list(self.getLegalActions(state))
        if len(actions) == 0:
            return None
        # Create a list of action and Q-values of all (state, action) pairs
        actionVal = [(action, self.getQValue(state, action))
                     for action in actions]
        # sort the array based on Qvalues in decreasing order
        actionVal.sort(key=lambda x: x[1], reverse=True)
        # filter the list to have elements with only the max Qvalue
        bestActions = list(
            filter(lambda x: x[1] == actionVal[0][1], actionVal))
        # return a random best action from all possible best actions
        return random.choice(bestActions)[0]

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        # if state is terminal state then return None
        # otherwise flip a coin with probability epsilon 
        # and return a random choice or best action based on 
        # the outcome of the coin toss
        if state == "TERMINAL_STATE" or legalActions == []:
            return None
        action = None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        # alpha and discount factor
        alpha = self.alpha
        Y = self.discount
        # update the Qvalue based on the formula
        self.Q[(state, action)] = (1 - alpha) * self.getQValue(
            state, action
        ) + alpha * (reward + Y * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        # calculating the Q-value as the dot product of feature vector and weight vector
        weights = self.getWeights()
        featureVector = self.featExtractor.getFeatures(state, action)
        return weights*featureVector

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        # alpha and discount factor
        alpha = self.alpha
        gamma = self.discount
        # calculating the diff and updating the weights based on formula 
        diff = (reward + gamma * self.getValue(nextState)) - self.getQValue(
            state, action)
        features = self.featExtractor.getFeatures(state, action)
        for i in features:
            self.weights[i] = self.weights[i] + alpha * diff * features[i]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
