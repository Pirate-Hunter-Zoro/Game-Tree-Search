#!/usr/bin/env python3

import math, random
from ...lib.game import discrete_soccer

RED_WIN = 500
BLUE_WIN = -500

def soccer(state, player_id):
    # The soccer evaluation function *must* look into the game state
    # when running. It will then return a number, where the larger the
    # number, the better the expected reward (or lower bound reward)
    # will be.
    #
    # For a good evaluation function, you will need to
    # SoccerState-specific information. The file
    # `src/lib/game/discrete_soccer.py` provides a description of all
    # useful SoccerState properties.
    if not isinstance(state, discrete_soccer.SoccerState):
        raise ValueError("Evaluation function incompatible with game type.")

    # We're going to evaluate the state of the player from the point of view of the red player.
    # That is, if the red player is the winner given this state, we'll return a super big POSITIVE number.
    # If the blue player is the winner given this state, we'll return a super big NEGATIVE number.
    # Otherwise, we'll need to dive into the game state and decide which player is at an advantage
    if state.reward(player_id=player_id) == 10:
        # Blue won
        return BLUE_WIN
    elif state.reward(player_id=player_id) == -10:
        # Blue lost
        return RED_WIN
    else:
        total = 0
        # The game is still ongoing

        # See who gets to go
        red_going = False
        if state.current_player_obj.team == discrete_soccer.Team.RED:
            red_going = True
        
        # See where the ball is and whose goal the ball is closer to
        x_ball = state.ball.x
        y_ball = state.ball.y
        x_red = state.players[0].x
        y_red = state.players[0].y
        x_blue = state.players[1].x
        y_blue = state.players[1].y
        level_with_goal = state.goal_bottom <= y_ball and y_ball <= state.goal_top
        
        return total