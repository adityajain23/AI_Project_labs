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
        # util.raiseNotDefined()
        # print(newPos)
        # print(newFood.asList())
        # for i in range(len(newGhostStates)):
        #     print(newGhostStates[i])
        # print(currentGameState.capsules)

        foodValue = 0
        foodRemaining = len(newFood)

        for food in newFood:
            foodValue += manhattanDistance(food, newPos)

        ghostValue = 0
        for ghost in newGhostStates:
            # if (ghost.scaredTimer>2):
            #     ghostValue+=100
            if (manhattanDistance(newPos, ghost.getPosition()) > 7):
                ghostValue += 50
            else:
                ghostValue += 6*manhattanDistance(newPos, ghost.getPosition())

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

        v = float("-inf")
        bestAction = gameState.getLegalActions(0)[0]
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            s = self.value(succ, 1, self.depth)
            if(v < s or v == float("-inf")):
                v = s
                bestAction = action

        return bestAction

        util.raiseNotDefined()

    def value(self, state, index, depth):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if index == 0:
            legalActions = state.getLegalActions(index)
            return max(self.value(state.generateSuccessor(0, action), 1, depth) for action in legalActions)
        else:
            legalActions = state.getLegalActions(index)
            if index == state.getNumAgents()-1:
                return min(self.value(state.generateSuccessor(index, action), 0, depth-1) for action in legalActions)
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

        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        bestAction = gameState.getLegalActions(0)[0]
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            s = self.value(succ, 1, self.depth-1, alpha, beta)
            if(v < s):
                v = s
                bestAction = action
        return bestAction

    def value(self, state, index, depth, alpha, beta):
        if(index == 0):
            v = float("-inf")
            legalActions = state.getLegalActions(index)
            # bestAction = legalActions[0]
            for action in legalActions:
                newState = state.generateSuccessor(index, action)
                if depth == 0 or newState.isWin() or newState.isLose():
                    newV = self.evaluationFunction(newState)
                else:
                    newV = self.value(newState, index+1, depth, alpha, beta)
                if v <= newV:
                    # bestAction = action
                    v = newV
                if (v > beta):
                    return v
                alpha = max(alpha, v)
        else:
            v = float("inf")
            legalActions = state.getLegalActions(index)
            # bestAction = legalActions[0]
            for action in legalActions:
                newState = state.generateSuccessor(index, action)
                if depth == 0 or newState.isWin() or newState.isLose():
                    newV = self.evaluationFunction(newState)
                else:
                    if index == state.getNumAgents()-1:
                        newV = self.value(newState, 0,
                                          depth-1, alpha, beta)
                    else:
                        newV = self.value(newState, index+1,
                                          depth, alpha, beta)
                if v >= newV:
                    # bestAction = action
                    v = newV
                if v < alpha:
                    return v
                beta = min(beta, v)
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
        v = float("-inf")
        bestAction = gameState.getLegalActions(0)[0]
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            s = self.value(succ, 1, self.depth)
            if(v < s or v == float("-inf")):
                v = s
                bestAction = action

        return bestAction

        util.raiseNotDefined()

    def value(self, state, index, depth):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if index == 0:
            legalActions = state.getLegalActions(index)
            return max(self.value(state.generateSuccessor(0, action), 1, depth) for action in legalActions)
        else:
            legalActions = state.getLegalActions(index)
            if index == state.getNumAgents()-1:
                return sum(self.value(state.generateSuccessor(index, action), 0, depth-1) for action in legalActions)/len(legalActions)
            else:
                return sum(self.value(state.generateSuccessor(index, action), index+1, depth) for action in legalActions)/len(legalActions)
        util.raiseNotDefined()


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

    foodValue = 0
    foodRemaining = len(newFood)

    for food in newFood:
        foodValue += manhattanDistance(food, newPos)

    ghostValue = 0
    for ghost in newGhostStates:
        # if (ghost.scaredTimer>2):
        #     ghostValue+=100
        if (manhattanDistance(newPos, ghost.getPosition()) > 10):
            ghostValue += 50
        else:
            ghostValue += 6*manhattanDistance(newPos, ghost.getPosition())

    value = ghostValue/50 + 8/(foodValue+1) + 3/(foodRemaining+1)

    return value
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
