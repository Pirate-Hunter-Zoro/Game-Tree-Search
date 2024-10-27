#!/usr/bin/env python3

import math, random
from ...lib.game import discrete_soccer

def soccer(state, player_id):
    # TODO: Implement this function!
    #
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

    # We're going to evaluate the state of the player from the point of view of the blue player.
    # That is, if the blue player is the winner given this state, we'll return a super big POSITIVE number.
    # If the red player is the winner given this state, we'll return a super big NEGATIVE number.
    # Otherwise, we'll need to dive into the game state and decide which player is at an advantage
    if state.reward(player_id=player_id) == 10:
        # Blue won
        return 200
    elif state.reward(player_id=player_id) == -10:
        # Blue lost
        return -200
    else:
        total = 0
        # The game is still ongoing

        # See who is in possession of the ball
        if state.current_player_obj.team == discrete_soccer.Team.BLUE:
            # Blue is happy to get to go
            total += 10
        else:
            # Red is happy to get to go
            total -= 10
        
        # See where the ball is and whose goal the ball is closer to
        x_ball = state.ball.x
        y_ball = state.ball.y
        level_with_goal = state.goal_bottom <= y_ball and y_ball <= state.goal_top
        # If the ball is closer to a certain goal AND level to be kicked into said goal, then SOMEONE is going to be really happy with the state
        ball_score_multiplier = 5 if level_with_goal else 1
        if abs(state.red_goal_pos - x_ball) < abs(state.blue_goal_pos - x_ball):
            # Ball is closer to the red goal than the blue goal, which makes the blue team happier the closer the ball is to the red goal
            total += abs(state.blue_goal_pos - x_ball) * ball_score_multiplier
        elif abs(state.red_goal_pos - x_ball) > abs(state.blue_goal_pos - x_ball):
            # Ball is closer to the blue goal, which makes the red ream happier the closer the ball is to the blue goal
            total -= abs(state.red_goal_pos - x_ball) & ball_score_multiplier

        # Now we should consider who is closer to being in possession of the ball
        if state.current_player_obj.team == discrete_soccer.Team.BLUE and state.current_player_obj.has_ball:
            # Blue team is quite happy to have the ball
            total += 75
        elif state.current_player_obj.team == discrete_soccer.Team.RED and state.current_player_ob.has_ball:
            # Red team is quite happy to have the ball
            total -= 75
        else:
            # See who is closer to the ball (Manhattan Distance)
            red_x = state.players[player_id].x \
                if state.players[player_id].team == discrete_soccer.Team.RED \
                else state.players[(player_id + 1) % 2].x
            blue_x = state.players[player_id].x \
                if state.players[player_id].team == discrete_soccer.Team.BLUE \
                else state.players[(player_id + 1) % 2].x
            red_y = state.players[player_id].y \
                if state.players[player_id].team == discrete_soccer.Team.RED \
                else state.players[(player_id + 1) % 2].y
            blue_y = state.players[player_id].y \
                if state.players[player_id].team == discrete_soccer.Team.BLUE \
                else state.players[(player_id + 1) % 2].y
            
            red_distance = abs(x_ball - red_x) + abs(y_ball - red_y)
            blue_distance = abs(x_ball - blue_x) + abs(y_ball - blue_y)
            # If blue is closer, score increases, and score decreases if red is closer
            total += 10 * (red_distance - blue_distance)

        return total