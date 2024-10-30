#!/usr/bin/env python3

from abc import abstractmethod
import math
import random

from src.lib.game.discrete_soccer import Action, GameState
from ...lib.game import Agent, RandomAgent
from ...lib.game._game import *
from typing import Self

print_trees = True

class BinaryGameTreeNode:

    __output_file = "Decision Tree.txt"

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
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
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

    # A class variable for deciding how many rollouts to apply to a child node
    __rollouts_before_expansion = 10

    def __init__(self, state: GameState, first_team: bool=True):
        """Constructor for the GameState object
        """
        super().__init__(state=state, first_team=first_team)
        self.__num_wins = 0
        self.__num_plays = 0

    def _get_key(self) -> str:
        return f'{self.__num_wins}/{self.__num_plays}'

    def __roll_out(self, seen_states: set[GameState]) -> bool:
        """Play out a game randomly from this node and return if a positive win occurs
        """
        if self._state in seen_states:
            # Repeat state for this particular rollout - tie game
            self.__num_plays += 1
            return False
        else:
            # Not a repeat state
            seen_states.add(self._state)
            if self.__num_plays >= MonteCarloNode.__rollouts_before_expansion:
                self.__expand(seen_states=seen_states)
            
            seen_states.add(self._state)
            if self._state.is_terminal:
                # See if this state ended in a win or loss
                self.__num_plays += 1
                old_wins = self.__num_wins
                self.__num_wins += 1 if self._state.reward(player_id = 0 if self._first_team else 1) > 0 else 0
                return self.__num_wins > old_wins
            else:
                # Pick a random child to roll out
                next_state = None
                options = random.sample(self._state.actions, len(self._state.actions))
                options_idx = 0
                while next_state == None:
                    next_state = self._state.act(options[options_idx])
                    options_idx += 1

                self.__num_plays += 1
                next_node = MonteCarloNode(next_state, first_team=self._first_team)
                if next_node.__roll_out(seen_states):
                    self.__num_wins += 1
                    return True
                else:
                    return False
        
    def __expand(self, seen_states: set[GameState]):
        """Helper method to give this Monte Carlo node child nodes
        """
        for action in self._state.actions:
            next_state = self._state.act(action)
            if next_state not in seen_states and next_state != None:
                # There's no point in adding a repeat state to explore - the game ends at that point
                self._children[next_state] = MonteCarloNode(state=next_state, first_team=self._first_team)
                self._actions[next_state] = action
                # Roll out this child node a few times
                self.__num_plays += MonteCarloNode.__rollouts_before_expansion
                for _ in range(MonteCarloNode.__rollouts_before_expansion):
                    if self._children[next_state].__roll_out(seen_states=seen_states):
                        self.__num_wins += 1

    # Helper variable to assist with heuristic
    __exploration_constant = 100

    def __monte_carlo_heuristic(self, child: Self) -> float:
        """Helper method to return a float representing how beneficial a particular node is for a heuristic to reach in Monte Carlo
        """
        return child.__num_wins / child.__num_plays + math.sqrt(MonteCarloNode.__exploration_constant*math.log(self.__num_plays)/child.__num_plays)

    def __traverse_monte_carlo(self, seen_states: set[GameState]):
        """Run the Monte Carlo algorithm from this tree node, which will use a heuristic to check which child node to make a recursive call on
        """
        if len(self._children) > 0:
            seen_states.add(self._state)
            # Look at the children, and based on our Monte Carlo heuristic, select the child to continue down
            best_child = self
            record_heuristic = float('-inf')
            for child in self._children.values():
                h = self.__monte_carlo_heuristic(child=child)
                if h > record_heuristic:
                    record_heuristic = h
                best_child = child
            old_best_child_plays = best_child.__num_plays
            old_best_child_wins = best_child.__num_wins
            # Make a recursive call on that child
            best_child.__traverse_monte_carlo(seen_states)
            # That could have led to the child or further descendant expanding and hence being played many times over
            self.__num_plays += best_child.__num_plays - old_best_child_plays
            self.__num_wins += best_child.__num_wins - old_best_child_wins
        else:
            # No children yet, so roll out
            self.__roll_out(seen_states=seen_states)

    # Class parameter to decide how many Monte Carlo repetitions shall be performed before a decision for the next move is made
    monte_carlo_preliminary_count = 100

    def decide(self) -> Action:
        """After performing Monte Carlo an adequate number of times, see what decision the algorithm will make
        """
        for _ in range(MonteCarloNode.monte_carlo_preliminary_count):
            self.__traverse_monte_carlo(seen_states=set())
        # Find the best child now that we have explored a bunch
        best_child = self
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
        MonteCarloNode.monte_carlo_preliminary_count = 100

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
        action = node.decide()
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

    # Class constant for the max depth we are willing to explore
    max_depth = 3

    # Keep track of each created node by its state to avoid repeats
    created_states = {}

    @staticmethod
    def make_game_node(state: GameState, maximizer: bool, depth: int, alpha_beta: bool) -> Self:
        """This helper method is to avoid creating repeat states and infinite loops with the preceding constructor
        """
        if state not in MinimaxNode.created_states.keys():
            return MinimaxNode(state=state, maximizer=maximizer, depth=depth, alpha_beta=alpha_beta)
        else:
            return None

    def __init__(self, state: GameState, maximizer: bool=True, depth: int=1, alpha_beta: bool=True):
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

        if depth < MinimaxNode.max_depth:
            # We keep growing the tree
            for action in self._state.actions:
                # The call to GameNode.make_game_node avoids infinite state creation repetition
                next_state = self._state.act(action=action)
                if next_state != None:
                    child_node = MinimaxNode.make_game_node(state=next_state, maximizer=not self._first_team, depth=depth+1, alpha_beta=self.__alpha_beta)
                    if child_node != None:
                        self._children[next_state] = child_node
                        self._actions[next_state] = action

    def _get_key(self) -> str:
        return str(self.__value)

    def __rec_get_minimax(self, alpha: float=float('-inf'), beta: float=float('inf'), depth: int=1) -> float:
        """Recursive helper method for returning a minimax value from a given node
        """
        if depth >= MinimaxNode.max_depth:
            return self.__value
        elif self.__alpha_beta:
            record_value = self.__value
            prune_these = []
            for state, child_node in self._children.items():
                prune = alpha >= beta
                if not prune:
                    value = child_node.__rec_get_minimax(alpha=alpha, beta=beta, depth=depth+1)
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
            return record_value
        else:
            # No alpha-beta pruning
            record_value = self.__value
            for _, child_node in self._children.items():
                value = child_node.__rec_get_minimax(depth=depth+1)
                if self._first_team and value > record_value:
                    # Maximizer and broke record
                    record_value = value
                elif (not self._first_team) and value < record_value:
                    # Minimizer and broke record
                    record_value = value

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
        MinimaxNode.max_depth = max_depth

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
        
        node = MinimaxNode(state=state, maximizer=(player == 0), depth=depth, alpha_beta=False)
        action = node.decide()
        if print_trees:
            node.display()
        return action

    def minimax_with_ab_pruning(self, state, player, depth=1,
                                alpha=float('inf'), beta=-float('inf')):

        node = MinimaxNode(state=state, maximizer=(player == 0), depth=depth, alpha_beta=True)
        action = node.decide()
        if print_trees:
            node.display()
        return action