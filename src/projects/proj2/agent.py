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

class BinaryGameTreeNode:

    def __init__(self, state: GameState):
        """Constructor for the GameState object
        """
        self._state = state
    
    @abstractmethod
    def decide(self) -> Action:
        pass

####################################################################################################
####################################################################################################
####################################################################################################
# Monte Carlo Implementation Follows...

class Outcome(Enum):
    DRAW = 0
    WIN = 1
    LOSS = 2

class MonteCarloNode(BinaryGameTreeNode):
    """Whenever we implement Monte-Carlo, we must have some kind of underlying tree.
    This class will be a node of that tree. 
    It will contain a game state.
    It will contain a the number of game rollouts played at the given state as well as the number of wins.
    It will also contain child nodes each with their own game states and values (unless this is a terminal state and hence results in a winning or losing game).
    """

    # Class variable to keep track of the states we have seen
    __seen_states = set()

    def __init__(self, state: GameState, parent: Self=None):
        """Constructor for the GameState object
        """
        super().__init__(state=state)
        self.__children = {}
        self.__actions = {}
        self.__num_wins = 0
        self.__num_plays = 0
        self.__parent = parent
        
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
        if len(self.__children) > 0:
            MonteCarloNode.__seen_states.add(self._state)
            # Look at the children, and based on our Monte Carlo heuristic, select the child to continue down
            best_child = None
            record_heuristic = float('-inf')
            for child in self.__children.values():
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
                next_node = MonteCarloNode(state=next_state, parent=self)
                # Any new node we create needs to be played once - otherwise our heuristic will have a division by zero error
                next_node.__simulate()
                self.__actions[next_state] = action
                self.__children[next_state] = next_node
    
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
        if current_state.reward(player_id=current_state.current_player) < 0:
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
        for child in self.__children.values():
            h = self.__monte_carlo_heuristic(child=child)
            if h > record_heuristic:
                record_heuristic = h
            best_child = child
        return self.__actions[best_child._state]

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

        return self.monte_carlo(state)

    def monte_carlo(self, state):
        # This is the suggested method you use to do MCTS.  Assume
        # `state` is the current state.
        node = MonteCarloNode(state=state, parent=None)
        action = node.decide(self.max_playouts)
        return action

####################################################################################################
####################################################################################################
####################################################################################################
# MINIMAX and ALPHA_BETA node implementations follows...

class MinimaxNode(BinaryGameTreeNode):

    def __init__(self, evaluation_function, player: int, state: GameState, seen_states_in_path: set[GameState]=set(), max_depth: int=5):
        """Constructor for the Minimax node object
        """
        maximizer = red_team(state=state, player_id=player)
        super().__init__(state=state)
        seen_states_in_path.add(state)
        self.__value = evaluation_function(self._state, player)
        self.__best_action = None

        if len(seen_states_in_path) <= max_depth:
            # We keep growing the tree
            record_output = float('-inf') if maximizer else float('inf')
            for action in self._state.actions:
                next_state = self._state.act(action=action)
                if next_state != None and next_state not in seen_states_in_path:
                    seen_states_copy = set()
                    for state in seen_states_in_path:
                        seen_states_copy.add(state)
                    # Create a child and see what it yields
                    opponent_player = (player + 1) % state.num_players
                    child = MinimaxNode(evaluation_function=evaluation_function, player=opponent_player, state=next_state, seen_states_in_path=seen_states_copy, max_depth=max_depth)
                    
                    # Update record
                    if maximizer:
                        # Maximizer
                        if record_output < child.__value:
                            record_output = child.__value
                            self.__best_action = action
                    else:
                        # Minimizer
                        if record_output > child.__value:
                            record_output = child.__value
                            self.__best_action = action
            self.__value = record_output

    def decide(self) -> Action:
        """Apply minimax algorithm to make a decision for the next move
        """
        if self.__best_action == None:
            return self._state.actions[random.randint(0, len(self._state.actions)-1)]
        return self.__best_action

class AlphaBetaNode(BinaryGameTreeNode):

    def __init__(self, evaluation_function, player: int, state: GameState, alpha: float=float('-inf'), beta: float=float('inf'), seen_states_in_path: set[GameState]=set(), max_depth: int=5):
        """Constructor for the Alpha Beta node object
        """
        maximizer = red_team(state=state, player_id=player)
        super().__init__(state=state)
        seen_states_in_path.add(state)
        self.__value = evaluation_function(self._state, player)
        self.__best_action = None

        if len(seen_states_in_path) <= max_depth:
            # We keep growing the tree
            record_output = float('-inf') if maximizer else float('inf')
            for action in self._state.actions:
                next_state = self._state.act(action=action)
                if next_state != None and next_state not in seen_states_in_path:
                    seen_states_copy = set()
                    for state in seen_states_in_path:
                        seen_states_copy.add(state)
                    # Create a child and see what it yields
                    opponent_player = (player + 1) % state.num_players
                    child = AlphaBetaNode(evaluation_function=evaluation_function, player=opponent_player, state=next_state, alpha=alpha, beta=beta, seen_states_in_path=seen_states_copy, max_depth=max_depth)
                    
                    # Update alpha or beta
                    if maximizer:
                        # Maximizer
                        alpha = max(alpha, child.__value)
                        if record_output < child.__value:
                            record_output = child.__value
                            self.__best_action = action
                        if alpha >= beta:
                            break
                    else:
                        # Minimizer
                        beta = min(beta, child.__value)
                        if record_output > child.__value:
                            record_output = child.__value
                            self.__best_action = action
                        if beta <= alpha:
                            break
            self.__value = record_output

    def decide(self) -> Action:
        """Apply minimax algorithm to make a decision for the next move
        """
        if self.__best_action == None:
            return self._state.actions[random.randint(0, len(self._state.actions)-1)]
        return self.__best_action
    
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

        opponent = (state.current_player + 1) % state.num_players
        if not self.alpha_beta_pruning:
            return self.minimax(state, opponent)
        else:
            return self.minimax_with_ab_pruning(state, opponent)

    def minimax(self, state, player):
        # This is the suggested method you use to do minimax.  Assume
        # `state` is the current state, `player` is the player that
        # the agent is representing (NOT the current player in
        # `state`!)
        
        node = MinimaxNode(evaluation_function=self.evaluate, player=player, state=state, max_depth=self.max_depth)
        action = node.decide()
        return action

    def minimax_with_ab_pruning(self, state, player,
                                alpha=float('inf'), beta=-float('inf')):

        node = AlphaBetaNode(evaluation_function=self.evaluate, player=player, state=state, alpha=alpha, beta=beta, max_depth=self.max_depth)
        action = node.decide()
        return action