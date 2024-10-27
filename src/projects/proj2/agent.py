#!/usr/bin/env python3

from io import _WrappedBuffer, TextIOWrapper
import math
from ...lib.game import Agent, RandomAgent
from ...lib.game._game import *
from typing import Self

class MonteCarloNode:
    """Whenever we implement Monte-Carlo, we must have some kind of underlying tree.
    This class will be a node of that tree. 
    It will contain a game state.
    It will contain a the number of game rollouts played at the given state as well as the number of wins.
    It will also contain child nodes each with their own game states and values (unless this is a terminal state and hence results in a winning or losing game).
    """
    
    # Keep track of created nodes by their states to avoid repeating node creations for the same state
    created_states = {}

    # Another class variable for deciding how many rollouts to apply to a child node
    __num_rollouts_for_expansion = 10

    @staticmethod
    def get_game_node(state: GameState, first_team: bool) -> Self:
        """This helper method is to avoid creating repeat states and infinite loops with the preceding constructor
        """
        if state in MonteCarloNode.created_states.keys():
            return MonteCarloNode.created_states[state]
        else:
            MonteCarloNode.created_states[state] = MonteCarloNode(state=state, first_team=first_team)

    def __init__(self, state: GameState, first_team: bool=True):
        """Constructor for the GameState object
        """
        self.__explored_in_monte_carlo = False
        self.__first_team = first_team
        self.__expanded = False
        self.__state = state
        self.__children = []
        for action in self.__state.actions:
            # The call to GameNode.get_game_node avoids infinite state creation repetition
            self.__children.append(MonteCarloNode.get_game_node(self.__state.act(action=action), first_team=self.__first_team))
        # In the case that this is the FIRST GameNode constructed, we need to add it to the static map
        if self.__state not in MonteCarloNode.created_states.keys():
            MonteCarloNode.created_states[self.__state] = self
        
        # The following will be useful for Monte Carlo
        self.__num_wins = 0
        self.__num_plays = 0

    __output_file_monte_carlo = "Monte Carlo.txt"

    def print_monte_carlo(self):
        """Helper method to display the Monte Carlo tree
        """
        with open(MonteCarloNode.__output_file_monte_carlo, "w") as f:
            self.__rec_print_monte_carlo(f, 0)
    
    def __rec_print_monte_carlo(self, file: TextIOWrapper[_WrappedBuffer], level: int=0):
        """Recursive part of the Monte Carlo tree display
        """
        file.write("  " * level + str(self.__num_wins) + '/' + str(self.__num_plays) + "\n")
        for child in self.__children:
            if child.__explored_in_monte_carlo:
                child.__rec_print_monte_carlo(file=file, level=level + 1)

    def __roll_out(self, seen_states: set[GameState]) -> bool:
        """Play out a game randomly from this node and return if a positive win occurs
        """
        if self.__state in seen_states:
            # Repeat state for this particular rollout - tie game
            self.__num_plays += 1
            return False
        else:
            # Not a repeat state
            seen_states.add(self.__state)
            if self.__state.is_terminal:
                # See if this state ended in a win or loss
                self.__num_plays += 1
                self.__num_wins += 1 if ((self.__value > 0 and self.__first_team) or (self.__value < 0 and (not self.__first_team))) else 0
                return self.__value > 0
            else:
                # Pick a random child to roll out
                idx = int(math.random()*len(self.__children))
                self.__num_plays += 1
                if self.__children[idx].roll_out(seen_states):
                    self.__num_wins += 1
                    return True
                else:
                    return False

    # Class constant to help with the Monte Carlo heuristic method
    __exploration_constant = 100

    def __monte_carlo_heuristic(self, child: Self) -> float:
        """Helper method to return a float representing how beneficial a particular node is for a heuristic to reach in Monte Carlo
        """
        return child.__num_wins / child.__num_plays + math.sqrt(MonteCarloNode.__exploration_constant*math.log(self.__num_plays)/child.__num_plays)

    def play_monte_carlo(self, seen_states: set[GameState]):
        """Run the Monte Carlo algorithm from this tree node, which will use a heuristic to check which child node to make a recursive call on
        """
        if not self.__explored_in_monte_carlo:
            self.__explored_in_monte_carlo = True
        if not self.__expanded:
            # Then roll this node out
            self.__num_plays += MonteCarloNode.__num_rollouts_for_expansion
            for _ in range(MonteCarloNode.__num_rollouts_for_expansion):
                self.__num_wins += 1 if self.__roll_out(seen_states=seen_states) else 0
            self.__expanded = True
        elif not self.__state.is_terminal:
            seen_states.add(self.__state)
            # Look at the children, and based on our Monte Carlo heuristic, select the child to continue down
            best_child = self
            record_heuristic = float('-inf')
            for child in self.__children:
                h = self.__monte_carlo_heuristic(child=child)
                if h > record_heuristic:
                    record_heuristic = h
                best_child = child
            old_best_child_plays = best_child.__num_plays
            old_best_child_wins = best_child.__num_wins
            best_child.play_monte_carlo(seen_states)
            self.__num_plays += best_child.__num_plays - old_best_child_plays
            self.__num_wins += best_child.__num_wins - old_best_child_wins

    # Class parameter to decide how many Monte Carlo repetitions are necessary before a decision for the next move is available
    monte_carlo_preliminary_count = 100

    def decide_monte_carlo(self) -> Self:
        """After performing Monte Carlo an adequate number of times, see what decision the algorithm will make
        """
        for _ in range(MonteCarloNode.__monte_carlo_preliminary_count):
            self.play_monte_carlo(seen_states=set())
        # Find the best child now that we have explored a bunch
        best_child = self
        record_heuristic = float('-inf')
        for child in self.__children:
            h = self.__monte_carlo_heuristic(child=child)
            if h > record_heuristic:
                record_heuristic = h
            best_child = child
        return best_child

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
        # TODO: Implement this agent!
        #
        # Read the documentation in /src/lib/game/_game.py for
        # information on what the decide function does.
        #
        # Do NOT call the soccer evaluation function that you write
        # directly from this function! Instead, use
        # `self.evaluate`. It will behave identically, but will be
        # able to work for multiple games.
        #
        # Do NOT call any SoccerState-specific functions! Assume that
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
        return super().decide(state)

####################################################################################################
####################################################################################################
####################################################################################################
# MINIMAX node implementation follows...

class MinimaxNode:

    # A class variable for evaluating this node's value
    evaluation_function = lambda state, id : state.reward(player_id=id)

    # Class constant for the max depth we are willing to explore
    max_depth = 5

    # Keep track of each created node by its state to avoid repeats
    created_states = {}

    @staticmethod
    def get_game_node(state: GameState, first_team: bool) -> Self:
        """This helper method is to avoid creating repeat states and infinite loops with the preceding constructor
        """
        if state in MinimaxNode.created_states.keys():
            return MinimaxNode.created_states[state]
        else:
            MinimaxNode.created_states[state] = MinimaxNode(state=state, first_team=first_team)

    def __init__(self, state: GameState, maximizer: bool=True):
        """Constructor for the GameState object
        """
        # TODO: figure out depth limit and boolean for pruning
        self.__is_maximizer = maximizer
        self.__value = MinimaxNode.evaluation_function(state=self.__state, id=self.__state.current_player)
        self.__state = state
        self.__children = []
        for action in self.__state.actions:
            # The call to GameNode.get_game_node avoids infinite state creation repetition
            self.__children.append(MinimaxNode.get_game_node(state=self.__state.act(action=action), maximizer=self.__is_maximizer))
        # In the case that this is the FIRST GameNode constructed, we need to add it to the static map
        if self.__state not in MinimaxNode.created_states.keys():
            MinimaxNode.created_states[self.__state] = self
        
    # The file a minimax tree will be outputted to when printed
    __output_file_minimax = "Minimax.txt"

    def print_minimax(self):
        """Helper method to display the Minimax tree
        """
        with open(MinimaxNode.__output_file_minimax, "w") as f:
            self.__rec_print_minimax(f, 0)
    
    def __rec_print_minimax(self, file: TextIOWrapper[_WrappedBuffer], level: int=0):
        """Recursive part of the Monte Carlo tree display
        """
        file.write("  " * level + str(self.__value) + "\n")
        for child in self.__children:
            child.__rec_print_minimax(file=file, level=level + 1)

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

        MonteCarloNode.evaluation_function = self.evaluate

    def decide(self, state):
        # TODO: Implement this agent!
        #
        # Read the documentation in /src/lib/game/_game.py for
        # information on what the decide function does.
        #
        # Do NOT call the soccer evaluation function that you write
        # directly from this function! Instead, use
        # `self.evaluate`. It will behave identically, but will be
        # able to work for multiple games.
        #
        # Do NOT call any SoccerState-specific functions! Assume that
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
        
        return super().decide(state)

    def minimax_with_ab_pruning(self, state, player, depth=1,
                                alpha=float('inf'), beta=-float('inf')):
        return super().decide(state)