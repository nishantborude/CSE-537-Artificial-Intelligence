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
    """
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from game import Directions
    from util import Stack
    #s = Directions.SOUTH;
    #w = Directions.WEST;
    path = []
    nodeStack = Stack()
    dirStack = Stack()
    visited = []
    fringeList = []
    nodeStack.push([problem.getStartState(), ''])
    while nodeStack.isEmpty() == False:
        Node = nodeStack.pop()
        visitingNode = Node[0]
        visitingDir = Node[1]
        # print "Poped:" , visitingNode 
        
        # check if node is successor of lastly visited node
        IsSuccessor = False
        if len(visited) > 0:
            for i in problem.getSuccessors(visited[-1]):
                if i[0] == visitingNode:
                    IsSuccessor = True;

            if IsSuccessor == False:
                print visitingNode, " Is not a successor of " , visited[-1]
                print "NEED TO GO BACK:" ,visitingNode 
                print path
                print visited
                print len(visited)
                i = len(visited) - 1
                print "successors of: " , visitingNode , "are"
                print problem.getSuccessors(visitingNode)
                SuccessorFound = False
                while i > 0 and SuccessorFound == False:

                    for vis in problem.getSuccessors(visitingNode):
                        if visited[i] == vis[0]:
                            SuccessorFound = True;
                            print "Successor ", visited[i] , "Found";
                    if SuccessorFound == False:
                        print visited[i]
                        print i, "Opposite of " , path[i-1] , '=' , Opposite(path[i-1])
                        path.append(Opposite(path[i-1]))
                        i-=1
                
        if visitingDir != '':
            path.append(visitingDir)
            print "Direction appended ", visitingDir
        """if dirStack.isEmpty() == False:
            visitingDir = dirStack.pop()
            path.append(visitingDir)
            print "Direction: ", visitingDir , " added"
        """
        visited.append(visitingNode)
        for i in reversed(problem.getSuccessors(visitingNode)):
            if i[0] not in visited and i[0] not in fringeList:
                #path.append(i[1])
                nodeStack.push([i[0],i[1]])
                #dirStack.push(i[1])
                fringeList.append(i[0])
                #print "pushing :" , i

    return path
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
