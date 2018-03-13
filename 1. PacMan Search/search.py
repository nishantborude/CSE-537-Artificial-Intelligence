# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def Opposite(directionStr):
    from game import Directions
    if directionStr == Directions.EAST:
        return Directions.WEST
    if directionStr == Directions.WEST:
        return Directions.EAST
    if directionStr == Directions.NORTH:
        return Directions.SOUTH
    if directionStr == Directions.SOUTH:
        return Directions.NORTH
    return Directions.STOP

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """ """
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # 
    # Following is iterative implementation of Depth First Search algorithm
    # 1. Push starting node on stack or goal is visited
    # 2. Until, stack is empty
    #    a. Pop from stack and mark it as visited
    #    b. add successor of unvisited popped node into stack
    #    
    from game import Directions
    from util import Stack
    nodeStack  = Stack()  # Create stack for holding exploration
    visited    = []       # To track visited nodes 
    #solutions = []       # Track all possible paths
    
    # Stack will hold node and it's direction list 
    nodeStack.push((problem.getStartState(), []))
    while nodeStack.isEmpty() == False:
        tempNode  = nodeStack.pop()
        node      = tempNode[0]  
        direction = tempNode[1]
        if node in visited:
            continue
        visited.append(node)    # Mark node visited
        if problem.isGoalState(node) == True: 
            # Path found
            # solutions.append(direction)
            # Comment out solutions to allow algorithm to 
            # search other least cost solutions.
            # But, this will not lead to optimal solution as
            # nodes are marked visited and does not allow algorithm to search
            # for other least cost path
            break

        for suc in reversed(problem.getSuccessors(node)):
            if suc[0] not in visited:
                nodeDir = list(direction)  # Give each successor separate copy of direction
                nodeDir.append(suc[1])
                nodeStack.push((suc[0], nodeDir))

    # return min(solutions)
    return direction

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    from game import Directions
    from util import Queue
    nodeQueue  = Queue()  # Create stack for holding exploration
    visited    = []       # To track visited nodes 
    
    # Queue will hold node and it's direction list 
    nodeQueue.push((problem.getStartState(), []))
    while nodeQueue.isEmpty() == False:
        tempNode  = nodeQueue.pop()
        node      = tempNode[0]  
        direction = tempNode[1]
        
        if node in visited:
            continue
        visited.append(node)    # Mark node visited

        if problem.isGoalState(node) == True: 
            # Path found
            break

        for suc in problem.getSuccessors(node):
            # Add successors in fringeList if not visited or already added
            if suc[0] not in visited:
                nodeDir = list(direction)  # Give each successor separate copy of direction
                nodeDir.append(suc[1])
                nodeQueue.push((suc[0], nodeDir))
    return direction


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from game import Directions
    from util import PriorityQueue
    nodePQueue  = PriorityQueue()  # Create stack for holding exploration
    visited    = []       # To track visited nodes 
    
    # Priority Queue will hold node and it's direction list 
    nodePQueue.push((problem.getStartState(), [], 0), 0)
    while nodePQueue.isEmpty() == False:
        tempNode  = nodePQueue.pop()
        node      = tempNode[0]  
        direction = tempNode[1]
        nodeCost  = tempNode[2]
        if node in visited:
            continue
        visited.append(node)    # Mark node visited
        if problem.isGoalState(node) == True: 
            # Path found
            break

        for suc in problem.getSuccessors(node):
            if suc[0] not in visited:
                nodeDir = list(direction)  # Give each successor separate copy of direction
                nodeDir.append(suc[1])
                nodePQueue.push((suc[0], nodeDir,nodeCost + suc[2]), nodeCost + suc[2])
    return direction



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    from game import Directions
    from util import PriorityQueue
    nodePQueue = PriorityQueue()  # Create stack for holding exploration
    visited    = []               # To track visited nodes 
    
    # Priority Queue will hold node and it's direction list 
    nodePQueue.push((problem.getStartState(), [], 0), 0)
    while nodePQueue.isEmpty() == False:
        tempNode  = nodePQueue.pop()
        node      = tempNode[0]  
        direction = tempNode[1]
        nodeCost  = tempNode[2]
        if node in visited:
            continue
        visited.append(node)    # Mark node visited
        if problem.isGoalState(node) == True: 
            # Path found
            break

        for suc in problem.getSuccessors(node):
            if suc[0] not in visited:
                nodeDir = list(direction)  # Give each successor separate copy of direction
                nodeDir.append(suc[1])
                nodePQueue.push((suc[0], nodeDir,nodeCost + suc[2]), nodeCost + suc[2] + heuristic(suc[0], problem))
    return direction


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
