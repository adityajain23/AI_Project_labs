# Group 8
# Team Members:
# Aditya Jain : 1903102
# Adwait Agashe: 1903103
# Gunjan Mayekar: 1903117

# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util
import math

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        if "Stop" in legalMoves:
            legalMoves.remove("Stop")
        # print(legalMoves)
        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Here we have used distance from ghost, distance from food, total food particles remaining
        # as features to evaluate our evaluation function.

        # Initialize values to use in evaluating value.
        foodValue = 0
        ghostValue = 0

        # Get the number of food particles remaining.
        foodRemaining = len(newFood)

        # Get sum of manhattan distances of all the food particles.
        for food in newFood:
            foodValue += manhattanDistance(food, newPos)

        # Loop for all new states of the ghosts
        for ghost in newGhostStates:

            # If ghost is at a manhattan distance more than 7, then we give less priority to get away
            # from the ghosts
            if (manhattanDistance(newPos, ghost.getPosition()) > 7):
                ghostValue += 50
            # If the ghost is closer, then we prioritize running away from the ghosts.
            else:
                ghostValue += 6*manhattanDistance(newPos, ghost.getPosition())

        # The return value will be a function of all the calculated values.
        value = ghostValue/50 + 8/(foodValue+1) + 3/(foodRemaining+1)

        return value


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Initializing best value and best action.
        v = float("-inf")
        bestAction = gameState.getLegalActions(0)[0]
        # Loop over all the possible actions from the root state.
        for action in gameState.getLegalActions(0):
            # Get the successor state from current state and action.
            succ = gameState.generateSuccessor(0, action)
            # Get value of current state.
            s = self.value(succ, 1, self.depth)
            # Update the best action and best value.
            if(v < s or v == float("-inf")):
                v = s
                bestAction = action
        # return the best action.
        return bestAction

    def value(self, state, index, depth):
        # If we reach at the max depth or we reach a terminal state,
        # then we return the utility of that state.
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        # If the current agent is the max agent then we return the maximum
        # possible value of all the successor states which can be reached from the
        # current state taking all possible legal actions.
        if index == 0:
            legalActions = state.getLegalActions(index)
            return max(self.value(state.generateSuccessor(0, action), 1, depth) for action in legalActions)
        # If the current agent is the min agent then we return the minimum possible value
        # of all the successor states which can be reached from the current state taking
        # all possible legal actions.
        else:
            legalActions = state.getLegalActions(index)

            # If the min agent is the last ghost, then we reduce the depth by 1 and update next index to 0
            if index == state.getNumAgents()-1:
                return min(self.value(state.generateSuccessor(index, action), 0, depth-1) for action in legalActions)
            # If the min agent is not the last ghost, then we keep constant depth and increase index by 1.
            else:
                return min(self.value(state.generateSuccessor(index, action), index+1, depth) for action in legalActions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Initializing best value, alpha, beta and best action.
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        bestAction = gameState.getLegalActions(0)[0]

        # Loop over all possible legal actions from the root state.
        for action in gameState.getLegalActions(0):

            # Get the successor state using current state and action.
            succ = gameState.generateSuccessor(0, action)

            # Get the value of the new state
            newV = self.value(succ, 1, self.depth, alpha, beta)
            # update best value and best action.
            if(v < newV or v == float("-inf")):
                v = newV
                bestAction = action
            # Update the value of alpha
            alpha = max(v, alpha)

            # Apply alpha beta pruning if condition is met
            if beta < alpha:
                break

        # return best action.
        return bestAction

    def value(self, state, index, depth, alpha, beta):
        # If we reach at the max depth or we reach a terminal state,
        # then we return the utility of that state.
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        # If the current agent is the max agent then we return the maximum
        # possible value of all the successor states which can be reached from the
        # current state taking all possible legal actions.
        elif index == 0:
            v = float("-inf")
            for action in state.getLegalActions(index):
                v = max(v, self.value(state.generateSuccessor(
                    index, action), 1, depth, alpha, beta))

                # Update value of alpha.
                alpha = max(v, alpha)
                # Apply alpha beta pruning if condition is met.
                if beta < alpha:
                    return v
            return v
        # If the current agent is the min agent then we return the minimum possible value
        # of all the successor states which can be reached from the current state taking
        # all possible legal actions.
        else:
            v = float("inf")
            for action in state.getLegalActions(index):
                # If the min agent is the last ghost, then we reduce the depth by 1 and update next index to 0
                if index == state.getNumAgents()-1:
                    v = min(v, self.value(state.generateSuccessor(
                        index, action), 0, depth-1, alpha, beta))

                # If the min agent is not the last ghost, then we keep constant depth and increase index by 1.
                else:
                    v = min(v, self.value(state.generateSuccessor(
                        index, action), index+1, depth, alpha, beta))
                # Update value of beta.
                beta = min(beta, v)
                # Apply alpha beta pruning if condition is met.
                if beta < alpha:
                    return v
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Initializing best value and best action.
        v = float("-inf")
        bestAction = gameState.getLegalActions(0)[0]
        # Loop over all the possible actions from the root state..
        for action in gameState.getLegalActions(0):
            # Get the successor state using current state and action.
            succ = gameState.generateSuccessor(0, action)
            # Get value of current state.
            s = self.value(succ, 1, self.depth)
            # Update the best action and best value.
            if(v < s or v == float("-inf")):
                v = s
                bestAction = action

        # return the best action.
        return bestAction

    def value(self, state, index, depth):

        # If we reach at the max depth or we reach a terminal state,
        # then we return the utility of that state.
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        # If the current agent is the max agent then we return the maximum
        # possible value of all the successor states which can be reached from the
        # current state taking all possible legal actions.
        if index == 0:
            legalActions = state.getLegalActions(index)
            return max(self.value(state.generateSuccessor(0, action), 1, depth) for action in legalActions)
        # If the current agent is the min agent then we return the average over all possible value
        # of the successor states which can be reached from the current state taking
        # all possible legal actions.
        else:
            legalActions = state.getLegalActions(index)
            # If the min agent is the last ghost, then we reduce the depth by 1 and update next index to 0
            if index == state.getNumAgents()-1:
                return sum(self.value(state.generateSuccessor(index, action), 0, depth-1) for action in legalActions)/len(legalActions)
            # If the min agent is not the last ghost, then we keep constant depth and increase index by 1.
            else:
                return sum(self.value(state.generateSuccessor(index, action), index+1, depth) for action in legalActions)/len(legalActions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [
        ghostState.scaredTimer for ghostState in newGhostStates]

    # Here we have used distance from ghost, distance from food, total food particles remaining
    # as features to evaluate our evaluation function.

    # Initialize values to use in evaluating value.
    foodValue = 0
    ghostValue = 0

    # Get the number of food particles remaining.
    foodRemaining = len(newFood)

    # Get sum of manhattan distances of all the food particles.
    for food in newFood:
        foodValue += manhattanDistance(food, newPos)

    # Loop for all new states of the ghosts
    for ghost in newGhostStates:

        # If ghost is at a manhattan distance more than 7, then we give less priority to get away
        # from the ghosts
        if (manhattanDistance(newPos, ghost.getPosition()) > 10):
            ghostValue += 50
        # If the ghost is closer, then we prioritize running away from the ghosts.
        else:
            ghostValue += 6*manhattanDistance(newPos, ghost.getPosition())

    # The return value will be a function of all the calculated values.
    value = ghostValue/60 + 8/(foodValue+1) + 3/(foodRemaining+1)

    return value


# Abbreviation
better = betterEvaluationFunction
