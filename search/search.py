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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    # Create the starting state of the problem (one node in structure, nothing visited)
    stack.push((problem.getStartState(), [],0))
    visited = set()

    # Keep visiting nodes until we have no unexplored node starts
    while not stack.isEmpty():
        state, moves, cost = stack.pop()

        if state not in visited:
            visited.add(state)
            # If the problem is the goal, we found it, return the list
            if problem.isGoalState(state):
                return moves
            # Add the new nodes to the list to explore
            for newState, newDir, newCost in problem.getSuccessors(state):
                stack.push((newState, moves + [newDir], cost + newCost))

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    # Create the starting state of the problem (one node in structure, nothing visited)
    queue.push((problem.getStartState(), [], 0))
    visited = set()

    # Keep visiting nodes until we have no unexplored node starts
    while not queue.isEmpty():
        state, moves, cost = queue.pop()

        if state not in visited:
            visited.add(state)
            # If the problem is the goal, we found it, return the list
            if problem.isGoalState(state):
                return moves
            # Add the new nodes to the list to explore
            for newState, newDir, newCost in problem.getSuccessors(state):
                queue.push((newState, moves + [newDir], cost + newCost))

    return []



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    "Search the node of least total cost first. "
    visited = set()
    p_queue = util.PriorityQueue()
    p_queue.push((problem.getStartState(), []), 0)

    while not p_queue.isEmpty():
        state, moves = p_queue.pop()

        if state not in visited:
            visited.add(state)
        else:
            continue

        if problem.isGoalState(state):
            return moves

        for successor, action, newCost in problem.getSuccessors(state):
            if successor not in visited:
                p_queue.push(
                    (successor, moves + [action]),
                    newCost + problem.getCostOfActions(moves))



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueueWithFunction(lambda newChange: newChange[-1] + heuristic(newChange[0], problem))
    return makeASearch(queue, problem)

def makeASearch(dataStructure, problem, heuristic=nullHeuristic):
    """
    :param dataStructure: a data structure to manage the fringe elements
    :param problem: The problem that must be solved
    :param heuristic: the heuristic by states in the problem will be judged
    :return: a list of directions to take to solve the problem
    """

    # Create the starting state of the problem (one node in structure, nothing visited)
    dataStructure.push((problem.getStartState(), [], heuristic(problem.getStartState(), problem)))
    visited = set()

    # Keep visiting nodes until we have no unexplored node starts
    while not dataStructure.isEmpty():
        state, moves, cost = dataStructure.pop()

        if state not in visited:
            visited.add(state)
            # If the problem is the goal, we found it, return the list
            if problem.isGoalState(state):
                return moves
            # Add the new nodes to the list to explore
            for newState, newDir, newCost in problem.getSuccessors(state):
                dataStructure.push((newState, moves + [newDir], cost + newCost))

    return []




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
