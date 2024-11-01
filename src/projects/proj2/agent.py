#!/usr/bin/env python3

from abc import abstractmethod
from enum import Enum
import math
import random

from src.lib.game.discrete_soccer import Action, GameState
from src.projects.proj2.evaluation import red_team
from ...lib.game import Agent, RandomAgent
from ...lib.game._game import *
from typing import Self

print_trees = True

class Outcome(Enum):
    DRAW = 0
    WIN = 1
    LOSS = 2

class BinaryGameTreeNode:

    __output_file = "game-tree.txt"

    def __init__(self, state: GameState, first_team: bool=True):
        """Constructor for the GameState object
        """
        self._first_team = first_team
        self._state = state
        self._num_wins = 0
        self._num_plays = 0

        # Map actions to the children they lead to
        self._children = {}

        # Map actions to the states they lead to
        self._actions = {}
    
    @abstractmethod
    def decide(self) -> Action:
        pass

    @abstractmethod
    def _get_key(self) -> str:
        pass

    def display(self):
        with open(BinaryGameTreeNode.__output_file, "w") as f:
            lines = self.__get_lines()
            for line in lines:
                f.write(line + '\n')
    
    def __get_lines(self) -> list[str]:
        """Returns list of strings - line by line - of this tree printed out"""
        s = self._get_key()
        sub_trees = [child.__get_lines() for child in self._children.values()]
        if len(sub_trees) == 0:
            # Base case
            return [f'  {s}  ']

        # Otherwise we need to think about how much space to graphically allocate for each child subtree
        max_width = max([len(sub_tree[-1]) for sub_tree in sub_trees])
        max_depth = max([len(sub_tree) for sub_tree in sub_trees])
        space = len(sub_trees) * max_width
        # The longest base layer of leaf nodes of all the subtrees will guarantee that we have adequate room to display all children
        top_white_space = space - len(s)
        lines = []

        # Center the key of our root node
        if top_white_space % 2:
            lines.append(" "*(top_white_space//2) + s + " "*(top_white_space//2 + 1))
        else:
            lines.append(" "*(top_white_space//2) + s + " "*(top_white_space//2))
        
        # Add a bar going down and a horizontal line to visually connect to children
        if len(lines[0]) % 2:
            lines.append(" "*(len(lines[0]) // 2) + "|" + " "*(len(lines[0]) // 2))
            lines.append(" " + "_"*(len(lines[0]) // 2 - 1) + "|" + "_"*(len(lines[0]) // 2 - 1) + " ")
        else:
            lines.append(" "*((len(lines[0])-1) // 2) + "|" + " "*(len(lines[0]) // 2))
            lines.append(" " + "_"*((len(lines[0])-1) // 2 - 1) + "|" + "_"*(len(lines[0]) // 2 - 1) + " ")

        children_lines = []
        bar_down_posns = []
        for i in range(max_depth):
            layer = ""
            for child_tree in sub_trees:
                if i < len(child_tree):
                    child_sub_tree_line = child_tree[i]
                    # Pad this line until it is the length of max_width
                    white_space = max_width - len(child_sub_tree_line)
                    if white_space % 2:
                        if i == 0:
                            bar_down_posns.append(len(layer) + white_space//2 + len(child_sub_tree_line)//2 + 1)
                        layer += " "*(white_space//2 + 1) + child_sub_tree_line + " "*(white_space//2)
                    else:
                        if i == 0:
                            bar_down_posns.append(len(layer) + white_space//2 + len(child_sub_tree_line)//2)
                        layer += " "*(white_space//2) + child_sub_tree_line + " "*(white_space//2)
                else:
                    # Then we just have a bunch of white space
                    layer += " "*max_width
            children_lines.append(layer)
        
        # Draw vertical connection to each child
        bars_down = " "*bar_down_posns[0] + "|"
        for idx_1, idx_2 in zip(bar_down_posns, bar_down_posns[1:]):
            bars_down += " "*(idx_2 - idx_1 - 1) + "|"
        bars_down += " "*(len(lines[0]) - len(bars_down))
        lines.append(bars_down)
        
        # Add all the children lines
        lines.extend(children_lines)

        return lines
    
class PrinterHelperNode(BinaryGameTreeNode):
    """This is a dummy game tree node class to make sure printing the node is working properly
    """
    def __init__(self, state: GameState, first_team: bool=True, depth: int=1):
        super().__init__(state=state, first_team=first_team)
        if depth < 2:
            for action in state.actions:
                new_state = state.act(action=action)
                if new_state != None:
                    self._children[new_state] = PrinterHelperNode(state=new_state, first_team=first_team, depth=depth + 1)

    def _get_key(self) -> str:
        return '1'
    
    def decide(self) -> Action:
        return None

####################################################################################################
####################################################################################################
####################################################################################################
# Monte Carlo Implementation Follows...

class MonteCarloNode(BinaryGameTreeNode):
    """Whenever we implement Monte-Carlo, we must have some kind of underlying tree.
    This class will be a node of that tree. 
    It will contain a game state.
    It will contain a the number of game rollouts played at the given state as well as the number of wins.
    It will also contain child nodes each with their own game states and values (unless this is a terminal state and hence results in a winning or losing game).
    """

    # Class variable to keep track of the states we have seen
    __seen_states = set()

    def __init__(self, state: GameState, parent: Self=None, first_team: bool=True):
        """Constructor for the GameState object
        """
        super().__init__(state=state, first_team=first_team)
        self.__num_wins = 0
        self.__num_plays = 0
        self.__parent = parent

    def _get_key(self) -> str:
        return f'{self.__num_plays - self.__num_wins}/{self.__num_plays}'
        
    # Helper variables to assist with heuristic
    __exploitation_constant = 100
    __exploration_constant = 100

    def __monte_carlo_heuristic(self, child: Self) -> float:
        """Helper method to return a float representing how beneficial a particular node is for a heuristic to reach in Monte Carlo
        """
        exploitation_bias = MonteCarloNode.__exploitation_constant*(1 - child.__num_wins/child.__num_plays)
        exploration_bias = MonteCarloNode.__exploration_constant*math.sqrt(math.log(self.__num_plays)/child.__num_plays)
        return exploitation_bias + exploration_bias

    def __select_leaf(self, root_node: bool=False) -> Self:
        """Run the Monte Carlo algorithm from this tree node, which will use a heuristic to check which child node to make a recursive call on
        """
        # We need to keep track of the states seen so far in the game so that when we eventually get to a leaf node we'll know what states to NOT repeat when said leaf node is simulated
        if root_node:
            MonteCarloNode.__seen_states = set()
        if len(self._children) > 0:
            MonteCarloNode.__seen_states.add(self._state)
            # Look at the children, and based on our Monte Carlo heuristic, select the child to continue down
            best_child = None
            record_heuristic = float('-inf')
            for child in self._children.values():
                h = self.__monte_carlo_heuristic(child=child)
                if h > record_heuristic:
                    record_heuristic = h
                best_child = child
            
            # Make a recursive call on that child
            return best_child.__select_leaf()
        else:
            # No children yet - we're already at the leaf
            return self

    def __expand(self):
        """Helper method to give this Monte Carlo node child nodes
        """
        for action in self._state.actions:
            next_state = self._state.act(action)
            if (next_state not in MonteCarloNode.__seen_states) and (next_state != None):
                # There's no point in adding a repeat state to explore - the game ends at that point
                MonteCarloNode.__seen_states.add(next_state)
                next_node = MonteCarloNode(state=next_state, parent=self, first_team=not self._first_team)
                # Any new node we create needs to be played once - otherwise our heuristic will have a division by zero error
                next_node.__simulate()
                self._actions[next_state] = action
                self._children[next_state] = next_node
    
    def __simulate(self):
        """Play out a game randomly from this node and return if a positive win occurs
        """
        # Keep choosing random actions until the game ends
        current_state = self._state
        self.__num_plays += 1
        while not current_state.is_terminal: 
            next_state = None
            options = random.sample(current_state.actions, len(current_state.actions))
            options_idx = -1
            while (next_state == None) or (next_state in MonteCarloNode.__seen_states):
                options_idx += 1
                if options_idx >= len(options):
                    # We were unable to continue from this state because we could go nowhere new
                    self.__back_propagate(outcome=Outcome.DRAW)
                    return
                else:
                    next_state = current_state.act(options[options_idx])
            MonteCarloNode.__seen_states.add(next_state)
            current_state = next_state

        # Now that the state is terminal, see if we won or lost
        loss = False
        player_id = 0 if self._first_team else 1
        if current_state.reward(player_id=player_id) < 0:
            loss = True
        self.__num_wins += 1 if loss else 0
        self.__back_propagate(outcome=Outcome.LOSS if loss else Outcome.WIN)
    
    def __back_propagate(self, outcome:Outcome):
        """Helper method to back-propagate the results of a simulation 
        """
        # We know our result - what does that mean for our parent?
        if self.__parent != None:
            self.__parent.__num_plays += 1
            self.__parent.__num_wins += 1 if outcome == Outcome.LOSS else 0
            self.__parent.__back_propagate(outcome=Outcome.LOSS if outcome==Outcome.WIN else (Outcome.WIN if outcome==Outcome.LOSS else Outcome.DRAW))

    def decide(self, num_simulations: int) -> Action:
        """Given the STARTING node of a game, perform the select->expand->simulate->propagate results and the make our choice
        """
        for _ in range(num_simulations):
            self.__select_leaf(root_node=True).__expand()
        # Find the best child now that we have explored a bunch
        best_child = None
        record_heuristic = float('-inf')
        for child in self._children.values():
            h = self.__monte_carlo_heuristic(child=child)
            if h > record_heuristic:
                record_heuristic = h
            best_child = child
        return self._actions[best_child._state]

####################################################################################################
# MONTE CARLO agent implementation follows

class MonteCarloAgent(RandomAgent):
    """An agent that makes decisions using Monte Carlo Tree Search (MCTS),
    using an evaluation function to approximately guess how good certain
    states are when looking far into the future.

    :param evaluation_function: The function used to make evaluate a
        GameState. Should have the parameters (state, player_id) where
        `state` is the GameState and `player_id` is the ID of the
        player to calculate the expected payoff for.

    :param max_playouts: The maximum number of playouts to perform
        using MCTS.
    """
    def __init__(self, evaluate_function, max_playouts=100):
        super().__init__()
        self.evaluate = evaluate_function
        self.max_playouts = max_playouts

    def decide(self, state):
        # Read the documentation in /src/lib/game/_game.py for
        # information on what the decide function does.
        #
        # Do NOT call the soccer evaluation function that you write
        # directly from this function! Instead, use
        # `self.evaluate`. It will behave identically, but will be
        # able to work for multiple games.
        #
        # Do NOT call any GameState-specific functions! Assume that
        # you can only see the functions provided in the GameState
        # class.
        #
        # If you would like to see some example agents, check out
        # `/src/lib/game/_agents.py`.

        return self.monte_carlo(state, state.current_player)

    def monte_carlo(self, state, player):
        # This is the suggested method you use to do MCTS.  Assume
        # `state` is the current state, `player` is the player that
        # the agent is representing (NOT the current player in
        # `state`!).
        node = MonteCarloNode(state=state, first_team=(player == 0))
        action = node.decide(self.max_playouts)
        if print_trees:
            node.display()
        return action

####################################################################################################
####################################################################################################
####################################################################################################
# MINIMAX node implementation follows...

class MinimaxNode(BinaryGameTreeNode):

    # A class variable for evaluating this node's value
    evaluation_function = lambda state, id : state.reward(player_id=id)

    # Keep track of each created node by its state to avoid repeats
    created_states = {}

    @staticmethod
    def make_game_node(state: GameState, maximizer: bool, depth: int, alpha_beta: bool, max_depth: int) -> Self:
        """This helper method is to avoid creating repeat states and infinite loops with the preceding constructor
        """
        if state not in MinimaxNode.created_states.keys():
            return MinimaxNode(state=state, maximizer=maximizer, depth=depth, alpha_beta=alpha_beta, max_depth=max_depth)
        else:
            return None

    def __init__(self, state: GameState, maximizer: bool=True, depth: int=1, alpha_beta: bool=True, max_depth: int=5):
        """Constructor for the GameState object
        """
        super().__init__(state=state, first_team=maximizer)
        self.__alpha_beta = alpha_beta
        if depth == 1:
            # New tree
            MinimaxNode.created_states = {}
        self.__value = MinimaxNode.evaluation_function(state=self._state, player_id=self._state.current_player)
        # We need to keep track of this state so that it will not repeat in any descendents
        MinimaxNode.created_states[self._state] = self

        if depth < max_depth:
            # We keep growing the tree
            for action in self._state.actions:
                # The call to GameNode.make_game_node avoids infinite state creation repetition
                next_state = self._state.act(action=action)
                if next_state != None:
                    child_node = MinimaxNode.make_game_node(state=next_state, maximizer=not self._first_team, depth=depth+1, alpha_beta=self.__alpha_beta, max_depth=max_depth)
                    if child_node != None:
                        self._children[next_state] = child_node
                        self._actions[next_state] = action

    def _get_key(self) -> str:
        return str(self.__value)

    def __rec_get_minimax(self, alpha: float=float('-inf'), beta: float=float('inf')) -> float:
        """Recursive helper method for returning a minimax value from a given node
        """
        if len(self._children) == 0:
            return self.__value
        elif self.__alpha_beta:
            record_value = None
            prune_these = []
            for state, child_node in self._children.items():
                prune = alpha >= beta
                if not prune:
                    value = child_node.__rec_get_minimax(alpha=alpha, beta=beta)
                    if record_value == None:
                        record_value = value
                    if self._first_team and value > record_value:
                        # Maximizer and broke record
                        record_value = value
                        alpha = max(alpha, record_value)
                    elif (not self._first_team) and value < record_value:
                        # Minimizer and broke record
                        record_value = value
                        beta = min(beta, record_value)
                else:
                    # From this point on, all children will be pruned
                    prune_these.append(state)
            for state in prune_these:
                del self._children[state]
            self.__value = record_value
            return record_value
        else:
            # No alpha-beta pruning
            record_value = None
            for _, child_node in self._children.items():
                value = child_node.__rec_get_minimax()
                if record_value == None:
                    record_value = value
                if self._first_team and value > record_value:
                    # Maximizer and broke record
                    record_value = value
                elif (not self._first_team) and value < record_value:
                    # Minimizer and broke record
                    record_value = value
            self.__value = record_value
            return record_value

    def decide(self) -> Action:
        """Apply minimax algorithm to make a decision for the next move
        """
        record_holder = self
        record_value = float('-inf') if self._first_team else float('inf')
        for _, child in self._children.items():
            value = child.__rec_get_minimax()
            if self._first_team and value > record_value:
                # Maximizer and broke record
                record_value = value
                record_holder = child
            elif (not self._first_team) and value < record_value:
                # Minimizer and broke record
                record_value = value
                record_holder = child

        return self._actions[record_holder._state]

####################################################################################################
# MINIMAX agent implementation follows...

class MinimaxAgent(RandomAgent):
    """An agent that makes decisions using the Minimax algorithm, using a
    evaluation function to approximately guess how good certain states
    are when looking far into the future.

    :param evaluation_function: The function used to make evaluate a
        GameState. Should have the parameters (state, player_id) where
        `state` is the GameState and `player_id` is the ID of the
        player to calculate the expected payoff for.

    :param alpha_beta_pruning: True if you would like to use
        alpha-beta pruning.

    :param max_depth: The maximum depth to search using the minimax
        algorithm, before using estimates generated by the evaluation
        function.
    """
    def __init__(self, evaluate_function, alpha_beta_pruning=False, max_depth=5):
        super().__init__()
        self.evaluate = evaluate_function
        self.alpha_beta_pruning = alpha_beta_pruning
        self.max_depth = max_depth

        MinimaxNode.evaluation_function = self.evaluate

    def decide(self, state):
        # Read the documentation in /src/lib/game/_game.py for
        # information on what the decide function does.
        #
        # Do NOT call the soccer evaluation function that you write
        # directly from this function! Instead, use
        # `self.evaluate`. It will behave identically, but will be
        # able to work for multiple games.
        #
        # Do NOT call any GameState-specific functions! Assume that
        # you can only see the functions provided in the GameState
        # class.
        #
        # If you would like to see some example agents, check out
        # `/src/lib/game/_agents.py`.

        if not self.alpha_beta_pruning:
            return self.minimax(state, state.current_player)
        else:
            return self.minimax_with_ab_pruning(state, state.current_player)

    def minimax(self, state, player, depth=1):
        # This is the suggested method you use to do minimax.  Assume
        # `state` is the current state, `player` is the player that
        # the agent is representing (NOT the current player in
        # `state`!)  and `depth` is the current depth of recursion.
        
        # Find out if player is red (maximizer) or blue
        node = MinimaxNode(state=state, maximizer=red_team(state, player), depth=depth, alpha_beta=False, max_depth=self.max_depth)
        action = node.decide()
        if print_trees:
            node.display()
        return action

    def minimax_with_ab_pruning(self, state, player, depth=1,
                                alpha=float('inf'), beta=-float('inf')):

        node = MinimaxNode(state=state, maximizer=(player == 0), depth=depth, alpha_beta=True, max_depth=self.max_depth)
        action = node.decide()
        if print_trees:
            node.display()
        return action