#!/usr/bin/env python3

import math

from src.lib.game.discrete_soccer import Action, GameState
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

    # A class variable for deciding how many rollouts to apply to a child node
    __rollouts_before_expansion = 10

    def __init__(self, state: GameState, first_team: bool=True):
        """Constructor for the GameState object
        """
        self.__first_team = first_team
        self.__state = state
        self.__num_wins = 0
        self.__num_plays = 0

        # Map actions to the children they lead to
        self.__children = {}

        # Map actions to the states they lead to
        self.__actions = {}

    __output_file_monte_carlo = "Monte Carlo.txt"

    def print_monte_carlo(self):
        """Helper method to display the Monte Carlo tree
        """
        with open(MonteCarloNode.__output_file_monte_carlo, "w") as f:
            self.__rec_print_monte_carlo(f, 0)
    
    def __rec_print_monte_carlo(self, file, level: int=0):
        """Recursive part of the Monte Carlo tree display
        """
        file.write("  " * level + str(self.__num_wins) + '/' + str(self.__num_plays) + "\n")
        for child in self.__children.items():
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
            if self.__num_plays >= MonteCarloNode.__rollouts_before_expansion:
                self.__expand(seen_states=seen_states)
            
            seen_states.add(self.__state)
            if self.__state.is_terminal:
                # See if this state ended in a win or loss
                self.__num_plays += 1
                old_wins = self.__num_wins
                self.__num_wins += 1 if self.__state.reward(player_id = 0 if self.__first_team else 1) > 0 else 0
                return self.__num_wins > old_wins
            else:
                # Pick a random child to roll out
                idx = int(math.random()*len(self.__state.actions))
                next_state = self.__state.act(self.__state.actions[idx])
                self.__num_plays += 1
                next_node = MonteCarloNode(next_state, first_team=self.__first_team)
                if next_node.roll_out(seen_states):
                    self.__num_wins += 1
                    return True
                else:
                    return False
        
    def __expand(self, seen_states: set[GameState]):
        """Helper method to give this Monte Carlo node child nodes
        """
        for action in self.__state.actions:
            next_state = self.__state.act(action)
            if next_state not in seen_states:
                # There's no point in adding a repeat state to explore - the game ends at that point
                self.__children[next_state] = MonteCarloNode(state=next_state, first_team=self.__first_team)
                self.__actions[next_state] = action
                # Roll out this child node a few times
                self.__num_plays += MonteCarloNode.__rollouts_before_expansion
                for _ in range(MonteCarloNode.__rollouts_before_expansion):
                    if self.__children[next_state].__roll_out(seen_states=seen_states):
                        self.__num_wins += 1

    def __monte_carlo_heuristic(self, child: Self) -> float:
        """Helper method to return a float representing how beneficial a particular node is for a heuristic to reach in Monte Carlo
        """
        return child.__num_wins / child.__num_plays + math.sqrt(MonteCarloNode.__exploration_constant*math.log(self.__num_plays)/child.__num_plays)

    def play_monte_carlo(self, seen_states: set[GameState]):
        """Run the Monte Carlo algorithm from this tree node, which will use a heuristic to check which child node to make a recursive call on
        """
        if len(self.__children) > 0:
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
            # Make a recursive call on that child
            best_child.play_monte_carlo(seen_states)
            # That could have led to the child or further descendant expanding and hence being played many times over
            self.__num_plays += best_child.__num_plays - old_best_child_plays
            self.__num_wins += best_child.__num_wins - old_best_child_wins
        else:
            # No children yet, so roll out
            self.__roll_out(seen_states=seen_states)

    # Class parameter to decide how many Monte Carlo repetitions shall be performed before a decision for the next move is made
    monte_carlo_preliminary_count = 100

    def decide_monte_carlo(self) -> Action:
        """After performing Monte Carlo an adequate number of times, see what decision the algorithm will make
        """
        for _ in range(MonteCarloNode.monte_carlo_preliminary_count):
            self.play_monte_carlo(seen_states=set())
        # Find the best child now that we have explored a bunch
        best_child = self
        record_heuristic = float('-inf')
        for child in self.__children:
            h = self.__monte_carlo_heuristic(child=child)
            if h > record_heuristic:
                record_heuristic = h
            best_child = child
        return self.__actions[best_child.__state]

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
        action = node.decide_monte_carlo()
        node.print_monte_carlo()
        return action

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
    def make_game_node(depth: int, state: GameState, maximizer: bool) -> Self:
        """This helper method is to avoid creating repeat states and infinite loops with the preceding constructor
        """
        if state not in MinimaxNode.created_states.keys():
            return MinimaxNode(depth=depth, state=state, maximizer=maximizer)
        else:
            return None

    def __init__(self, depth: int, state: GameState, maximizer: bool=True):
        """Constructor for the GameState object
        """
        if depth == 1:
            # New tree
            MinimaxNode.created_states = {}
        self.__is_maximizer = maximizer
        self.__state = state
        self.__value = MinimaxNode.evaluation_function(state=self.__state, player_id=self.__state.current_player)
        # We need to keep track of this state so that it will not repeat in any descendents
        MinimaxNode.created_states[self.__state] = self
        # We need the children to be a dictionary so they can be pruned
        self.__children = {}
        # We also need the actions that correspond to each respective child state
        self.__actions = {}

        if depth < MinimaxNode.max_depth:
            # We keep growing the tree
            for action in self.__state.actions:
                # The call to GameNode.make_game_node avoids infinite state creation repetition
                next_state = self.__state.act(action=action)
                if next_state != None:
                    child_node = MinimaxNode.make_game_node(depth=depth+1, state=next_state, maximizer=not self.__is_maximizer)
                    if child_node != None:
                        self.__children[next_state] = child_node
                        self.__actions[next_state] = action
        
    # The file a minimax tree will be outputted to when printed
    __output_file_minimax = "Minimax.txt"

    def print_minimax(self):
        """Helper method to display the Minimax tree
        """
        with open(MinimaxNode.__output_file_minimax, "w") as f:
            self.__rec_print_minimax(f, 0)
    
    def __rec_print_minimax(self, file, level: int=0):
        """Recursive part of the Monte Carlo tree display
        """
        if level <= MinimaxNode.max_depth:
            file.write("  " * level + str(self.__value) + "\n")
            for child in self.__children.values():
                child.__rec_print_minimax(file=file, level=level + 1)

    def __rec_get_minimax(self, alpha_beta: bool, alpha: float=float('-inf'), beta: float=float('inf'), depth: int=1) -> float:
        """Recursive helper method for returning a minimax value from a given node
        """
        if depth >= MinimaxNode.max_depth:
            return self.__value
        elif alpha_beta:
            record_value = self.__value
            prune_these = []
            for state, child_node in self.__children.items():
                prune = alpha >= beta
                if not prune:
                    value = child_node.__rec_get_minimax(alpha_beta=alpha_beta, alpha=alpha, beta=beta, depth=depth+1)
                    if self.__is_maximizer and value > record_value:
                        # Maximizer and broke record
                        record_value = value
                        alpha = max(alpha, record_value)
                    elif (not self.__is_maximizer) and value < record_value:
                        # Minimizer and broke record
                        record_value = value
                        beta = min(beta, record_value)
                else:
                    prune_these.append(state)
            for state in prune_these:
                del self.__children[state]
            return record_value
        else:
            # No alpha-beta pruning
            record_value = self.__value
            for _, child_node in self.__children.items():
                value = child_node.__rec_get_minimax(alpha_beta=alpha_beta, depth=depth+1)
                if self.__is_maximizer and value > record_value:
                    # Maximizer and broke record
                    record_value = value
                elif (not self.__is_maximizer) and value < record_value:
                    # Minimizer and broke record
                    record_value = value

            return record_value

    def decide_minimax(self, alpha_beta: bool=True, alpha: float=float('-inf'), beta: float=float('inf'), depth: int=1) -> Action:
        """Apply minimax algorithm to make a decision for the next move
        """
        record_holder = self
        record_value = alpha if self.__is_maximizer else beta
        for _, child in self.__children.items():
            value = child.__rec_get_minimax(alpha_beta=alpha_beta, alpha=alpha, beta=beta, depth=depth)
            if self.__is_maximizer and value > record_value:
                # Maximizer and broke record
                record_value = value
                record_holder = child
            elif (not self.__is_maximizer) and value < record_value:
                # Minimizer and broke record
                record_value = value
                record_holder = child

        return self.__actions[record_holder.__state]

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
        
        node = MinimaxNode(depth=depth, state=state, maximizer=(player == 0))
        action = node.decide_minimax(alpha_beta=False, depth=depth)
        node.print_minimax()
        return action

    def minimax_with_ab_pruning(self, state, player, depth=1,
                                alpha=float('inf'), beta=-float('inf')):

        node = MinimaxNode(depth=depth, parent_state=None, state=state, maximizer=(player == 0))
        action = node.decide_minimax(alpha_beta=True, alpha=alpha, beta=beta, depth=depth)
        node.print_minimax()
        return action