#!/usr/bin/env python3

import math, random
from ...lib.game import discrete_soccer

RED_WIN = float('inf')
BLUE_WIN = float('-inf')

def red_team(state, player_id) -> bool:
    """Helper method to determine the team of the player
    """
    return state.players[player_id].team == discrete_soccer.Team.RED

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
    if state.reward(player_id=player_id) == 10 or state.reward(player_id=player_id) == -10:
        # Someone won
        if state.winner == discrete_soccer.Team.RED:
            return RED_WIN
        else:
            return BLUE_WIN
    else:
        # The game is still ongoing
        score = 0

        # Constant max distance between any two points
        max_distance = state.pitch.width + state.pitch.height + 2
        
        # Grab the player from whose POV we are calculating the score
        player = state.players[player_id]

        # Reward the player for having the ball if they have it
        if player.has_ball:
            score += 1000

        # Grab the opponent player
        opponent = state.players[(player_id + 1) % len(state.players)]
        
        # See where the ball is and whose goal the ball is closer to
        x_ball = state.ball.x
        y_ball = state.ball.y
        x_player = player.x
        x_opponent = opponent.x
        y_player = player.y
        y_opponent = opponent.y

        # See where the each goal is - the built-in distance function to the goals is not appearing to be helpful...
        x_goal_opponent = state.red_goal_pos[0] if opponent.team == discrete_soccer.Team.RED else state.blue_goal_pos[0]
        y_goal_opponent = state.red_goal_pos[1] if opponent.team == discrete_soccer.Team.RED else state.blue_goal_pos[1]
        x_goal_player = state.red_goal_pos[0] if player.team == discrete_soccer.Team.RED else state.blue_goal_pos[0]
        y_goal_player = state.red_goal_pos[1] if player.team == discrete_soccer.Team.RED else state.blue_goal_pos[1]

        # Calculate players' distances to ball
        player_ball_distance = abs(x_ball - x_player) + abs(y_ball - y_player)
        opponent_ball_distance = abs(x_ball - x_opponent) + abs(y_ball - y_opponent)

        # Reward the player for being closer to the ball
        ball_distance_weight = 500
        score += ball_distance_weight * (max_distance - player_ball_distance)

        # The following weight will be useful in the case of ball possessions
        movement_weight = 1000

        # We want a few distances
        dist_player_goal_player = abs(player.x - x_goal_player) + abs(player.y - y_goal_player)
        dist_player_goal_opponent = abs(player.x - x_goal_opponent) + abs(player.y - y_goal_opponent)
        dist_between_players = abs(player.x - opponent.x) + abs(player.y - opponent.y)

        # If the opponent is CLOSER to the ball than we are, GET TOWARDS OUR GOAL SO WE CAN PROTECT IT
        if opponent_ball_distance <= player_ball_distance:
            score += movement_weight * (max_distance - dist_player_goal_player)

        # If the opponent HAS the ball, double the need to return to our own goal
        if opponent.has_ball:
            score += movement_weight * (max_distance - dist_player_goal_player)

        # On the other hand, if we have possession
        if player.has_ball:
            # Encourage movement away from the enemy player
            score += dist_between_players * movement_weight
            # If we have the ball, we'd like to be far away from our goal
            score += movement_weight * dist_player_goal_player
            # We'd also like to be closer to the enemy goal
            score += movement_weight * (max_distance - dist_player_goal_opponent)

        # We should also look ahead into the future - whatever state we end up at, the OPPONENT is going to go next
        if opponent_ball_distance == 1 and player_ball_distance > 1:
            # We can't stop the opponent from getting the ball - BEE LINE FOR OUR GOAL
            score += movement_weight * (max_distance - dist_player_goal_player)
        
        # If the player is the minimizer - or blue team - return the OPPOSITE of the score
        return score if red_team(state, player_id) else -score