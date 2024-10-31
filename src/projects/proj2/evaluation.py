#!/usr/bin/env python3

import math, random
from ...lib.game import discrete_soccer

RED_WIN = 1000
BLUE_WIN = -1000

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
        
        # Grab the player from whose POV we are calculating the score
        player = state.players[player_id]

        # Reward the player for having the ball if they have it
        if player.has_ball:
            score += 300

        # Grab the opponent player
        opponent = state.players[(player_id + 1) % len(state.players)]
        
        # See where the ball is and whose goal the ball is closer to
        x_ball = state.ball.x
        y_ball = state.ball.y
        x_player = player.x
        x_opponent = opponent.x
        y_player = player.y
        y_opponent = opponent.y

        # Calculate players' distances to ball
        player_ball_distance = abs(x_ball - x_player) + abs(y_ball - y_player)
        opponent_ball_distance = abs(x_ball - x_opponent) + abs(y_ball - y_opponent)

        # Punish the player for being away from the ball
        ball_distance_weight = 10
        score -= player_ball_distance * ball_distance_weight

        # Reward the player for the opponent being away from the ball
        score += opponent_ball_distance * ball_distance_weight

        # The following weight will be useful in the case of ball possessions
        ball_possession_weight = 20

        # We want four distances
        dist_player_goal_player = state.dist_to_goal((player.x, player.y), player.team)
        dist_player_goal_opponent = state.dist_to_goal((player.x, player.y), opponent.team)
        dist_opponent_goal_player = state.dist_to_goal((opponent.x, opponent.y), player.team)
        dist_opponent_goal_opponent = state.dist_to_goal((opponent.x, opponent.y), opponent.team)

        # If the opponent has the ball, maybe we should advance, or maybe we should defend
        if opponent.has_ball:
            # If the player is closer to their own goal than the opponent is, then it's safe to approach the opponent
            if dist_player_goal_player < dist_opponent_goal_player:
                # Punish them for NOT approaching the opponent
                score -= ball_possession_weight * player_ball_distance
            else:
                # BAD BAD BAD - we do NOT want the opponent closer to our goal than we are WITH the ball
                # GET CLOSE TO OUR GOAL NOW
                score -= (ball_possession_weight)**2 * dist_player_goal_player

        # On the other hand, we have the reverse situation if we have possession
        if player.has_ball:
            # If the opponent is closer to their goal than the player is, there's  no point in approaching
            if dist_opponent_goal_opponent < dist_player_goal_opponent:
                # Reward the player for NOT approaching the opponent
                score += ball_possession_weight * (abs(player.x - opponent.x) + abs(player.y - opponent.y))
            else:
                # GOOD GOOD GOOD - we would LOVE to be closer to the opponent's goal than the opponent when we have the ball
                # Encourage movement farther away from our goal
                score += (ball_possession_weight)**2 * dist_player_goal_player
        
        # If the player is the minimizer - or blue team - return the OPPOSITE of the score
        return score if red_team(state, player_id) else -score