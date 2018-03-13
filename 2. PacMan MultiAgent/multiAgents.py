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
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newFood = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if newFood:
            fsum = 0
            gsum = 0            
            foodDist = []
            ghostDist = []
            for f in newFood:  
                foodDist.append(manhattanDistance(newPos,f))
            for g in newGhostStates:
                ghostDist.append(manhattanDistance(newPos,g.getPosition()))
            return successorGameState.getScore() + min(ghostDist) - min(foodDist)
        return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

#nodesExpanded = 0
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
        
    """
    #nodesExpanded = 0

    def maxMinimax(self, state, currentAgent, num, d):
        from game import Directions
        negINF = -9999.99
        # If current state is WIN or LOSE, do not evaluate its successors.
        if state.isWin() or state.isLose() or d == 0: 
            return self.evaluationFunction(state)

        moves = state.getLegalActions(0)
        maxScore = negINF
        maxAction = Directions.STOP
        for action in moves:
            #self.nodesExpanded += 1
            score = self.minMinimax(state.generateSuccessor(0, action), 1, num, d)
            if score > maxScore or (score == maxScore and maxAction == Directions.STOP):
                maxScore = score
                maxAction = action

        if d == self.depth:
            return maxAction
        return maxScore
         
    def minMinimax(self, state, currentAgent, num, d):
        score = []          
        
        # If current state is WIN or LOSE, do not evaluate its successors.
        if state.isWin() or state.isLose() or d == 0:
            return self.evaluationFunction(state)
        
        moves = state.getLegalActions(currentAgent) 
        if currentAgent < num:
            for action in moves:
                #self.nodesExpanded += 1
                score.append(self.minMinimax(state.generateSuccessor(currentAgent, action), currentAgent+1, num, d))
        else:
            for action in moves:
                #self.nodesExpanded += 1
                score.append(self.maxMinimax(state.generateSuccessor(currentAgent, action), currentAgent,num, d-1))
        return min(score)
        
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
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        d = self.depth
        action = self.maxMinimax(gameState, 0, numAgents-1, d)
        #print self.nodesExpanded
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
                                :DESCRIPTION:
        getAction() will start with max Node for deciding upon it's next move
        Max Node internally calls Min Node which returns the minimum score
        Alpha Beta pruning is used to reduce number of nodes expanded.
    """
    
    nodesExpanded = 0
    def maxMinimax(self, state, alpha, beta, currentAgent, num, d):
        # Evaluation done at Max Node
        # Calls it's successor min node and updates alpha value 
        # If Max value is greater than Beta then does not expand node further
        from game import Directions
        negINF = -9999

        # If current state is WIN or LOSE, do not evaluate its successors.
        if state.isWin() or state.isLose() or d == 0: 
            return state.getScore()

        moves = state.getLegalActions(0)
        maxScore = negINF
        maxAction = Directions.STOP

        for action in moves:
            #self.nodesExpanded += 1
            score = self.minMinimax(state.generateSuccessor(0, action), alpha, beta, 1, num, d)
            if score > maxScore or (score == maxScore and maxAction == Directions.STOP):
                maxScore = score
                maxAction = action
            alpha = max(alpha, maxScore)
            if maxScore > beta:
                if d == self.depth:
                    return maxAction
                return maxScore
        if d == self.depth:
            return maxAction
        else:
            return maxScore
         
    def minMinimax(self, state, alpha, beta,currentAgent, num, d):
        # Evaluation done at Min Node
        # Calls min node evaluation function for all agents and updates beta value
        # Once done, calls successor Max evaluation function
        # If minimum value is less than Alpha then does not expands nodes further
        score  = 0
        posINF = 9999

        # If current state is WIN or LOSE, do not evaluate its successors.
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        moves    = state.getLegalActions(currentAgent) 
        minScore = posINF
        for action in moves:
            #self.nodesExpanded += 1
            if d == 0:
                score = self.evaluationFunction(state.generateSuccessor(currentAgent, action))
            elif currentAgent < num:
                score = self.minMinimax(state.generateSuccessor(currentAgent, action), alpha, beta, currentAgent+1, num, d)
            else:
                score = self.maxMinimax(state.generateSuccessor(currentAgent, action), alpha, beta, currentAgent,num, d-1)
            if score < minScore:
                minScore = score
            beta = min(beta, minScore)
            if minScore < alpha:
                return minScore

        return minScore
 
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        d         = self.depth
        INF       = 99999
        alpha     = -INF
        beta      = INF
        action    = self.maxMinimax(gameState, alpha, beta, 0, numAgents-1, d)
        #print self.nodesExpanded
        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
                               :DESCRIPTION:
        getAction() will start with max Node for deciding upon it's next move
        Max Node internally calls Case Node which averages the expected score
    """
    #nodesExpanded = 0
    def maxExpectimax(self, state, currentAgent, num, d):
        from game import Directions
        negINF = -9999.99
        if state.isWin() or state.isLose() or d == 0:
            return self.evaluationFunction(state)
        moves = state.getLegalActions(0)
        maxScore = negINF
        maxAction = Directions.STOP
        for action in moves:
            #self.nodesExpanded += 1
            score = self.caseExpectimax(state.generateSuccessor(0, action), 1, num, d)
            if score > maxScore or (score == maxScore and maxAction == Directions.STOP):
                maxScore = score
                maxAction = action

        if d == self.depth:
            return maxAction
        return maxScore
         
    def caseExpectimax(self, state, currentAgent, num, d):
        if state.isWin() or state.isLose() or d == 0:
            return self.evaluationFunction(state)
      
        score = 0.0
        moves = state.getLegalActions(currentAgent) 

        for action in moves:
            #self.nodesExpanded += 1
            if d == 0:
                score += self.evaluationFunction(state.generateSuccessor(currentAgent, action))
            elif currentAgent < num:
                score += self.caseExpectimax(state.generateSuccessor(currentAgent, action), currentAgent+1, num, d)
            else:
                score += self.maxExpectimax(state.generateSuccessor(currentAgent, action), currentAgent,num, d-1)
        return score/len(moves) 

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        d = self.depth
        action = self.maxExpectimax(gameState, 0, numAgents-1, d)
        #print self.nodesExpanded
        return action
 

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

