import sys
import numpy
import random


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Game:

    def __init__(self):
        self.REWARD_FOUND_GOAL = 100
        self.REWARD_CAUGHT = -100
        self.REWARD_TIME_EXPIRED = 0
        self.REWARD_OTHER = -1

        self.MAX_STEPS = 50

        self.MIN_ACTION = 0
        self.MAX_ACTION = 4
        self.ACTION_NONE = 0
        self.ACTION_LEFT = 1
        self.ACTION_RIGHT = 2
        self.ACTION_UP = 3
        self.ACTION_DOWN = 4

        self.GOAL_X = 6
        self.GOAL_Y = 6

        self.num_steps = 1


def setupMaze():
    maze = numpy.ones((8, 8), dtype=int)
    for x in range(2, 7):
        for y in range(2, 7):
            maze[x, y] = 0
    maze[4, 4] = 1
    return maze


def initialize(game, maze, player, opponent):
    game.num_steps = 1
    while True:
        player.x = random.choice([2, 3, 4, 5, 6])
        player.y = random.choice([2, 3, 4, 5, 6])
        if (maze[player.x, player.y] == 0) and ((player.x != game.GOAL_X) or (player.y != game.GOAL_Y)):
            break
    while True:
        opponent.x = random.choice([2, 3, 4, 5, 6])
        opponent.y = random.choice([2, 3, 4, 5, 6])
        if (maze[opponent.x, opponent.y] == 0) and ((opponent.x != player.x) or (opponent.y != player.y)):
            break


def make_player_move(game, maze, player, opponent, action):
    if action == game.ACTION_NONE:
        pass
    elif action == game.ACTION_LEFT:
        if maze[player.x - 1, player.y] == 0:
            player.x -= 1
    elif action == game.ACTION_RIGHT:
        if maze[player.x + 1, player.y] == 0:
            player.x += 1
    elif action == game.ACTION_UP:
        if maze[player.x, player.y - 1] == 0:
            player.y -= 1
    elif action == game.ACTION_DOWN:
        if maze[player.x, player.y + 1] == 0:
            player.y += 1
    else:
        print(f'Unknown action {action}')

    if (player.x == game.GOAL_X) and (player.y == game.GOAL_Y):
        return 1
    if (player.x == opponent.x) and (player.y == opponent.y):
        return -1
    return 0


def opponent_move(game, maze, player, opponent):
    best_action = game.ACTION_NONE
    best_dist = ((player.x - opponent.x) ** 2) + ((player.y - opponent.y) ** 2)
    if maze[opponent.x - 1, opponent.y] == 0:
        the_dist = ((player.x - (opponent.x - 1)) ** 2) + ((player.y - opponent.y) ** 2)
        if the_dist < best_dist:
            best_action = game.ACTION_LEFT
            best_dist = the_dist
    if maze[opponent.x + 1, opponent.y] == 0:
        the_dist = ((player.x - (opponent.x + 1)) ** 2) + ((player.y - opponent.y) ** 2)
        if the_dist < best_dist:
            best_action = game.ACTION_RIGHT
            best_dist = the_dist
    if maze[opponent.x, opponent.y - 1] == 0:
        the_dist = ((player.x - opponent.x) ** 2) + ((player.y - (opponent.y - 1)) ** 2)
        if the_dist < best_dist:
            best_action = game.ACTION_UP
            best_dist = the_dist
    if maze[opponent.x, opponent.y + 1] == 0:
        the_dist = ((player.x - opponent.x) ** 2) + ((player.y - (opponent.y + 1)) ** 2)
        if the_dist < best_dist:
            best_action = game.ACTION_DOWN
            best_dist = the_dist

    if best_action == game.ACTION_NONE:
        pass
    elif best_action == game.ACTION_LEFT:
        opponent.x -= 1
    elif best_action == game.ACTION_RIGHT:
        opponent.x += 1
    elif best_action == game.ACTION_UP:
        opponent.y -= 1
    elif best_action == game.ACTION_DOWN:
        opponent.y += 1

    if best_dist == 0.0:
        return 1
    return 0


def execute_action(game, maze, player, opponent, action):
    move_result = make_player_move(game, maze, player, opponent, action)
    if move_result == 1:
        return game.REWARD_FOUND_GOAL
    elif move_result == -1:
        return game.REWARD_CAUGHT
    game.num_steps += 1
    if opponent_move(game, maze, player, opponent) == 1:
        return game.REWARD_CAUGHT
    if game.num_steps >= game.MAX_STEPS:
        return game.REWARD_TIME_EXPIRED
    return game.REWARD_OTHER


def show_maze(game, maze, player, opponent):
    for y in range(1, 8):
        for x in range(1, 8):
            if maze[x, y] == 1:
                print("X", end="")
            elif (x == player.x) and (y == player.y):
                print("P", end="")
            elif (x == opponent.x) and (y == opponent.y):
                print("O", end="")
            elif (x == game.GOAL_X) and (y == game.GOAL_Y):
                print("G", end="")
            else:
                print(" ", end="")
        print()


def train_pick_action(game, maze, player, opponent):
    # Add code here to construct a state number from the state information
    #   (player_point and opponent_point)
    # Use your state to pick an action by looking up the value of actions
    #   for your state, for some large percentage of the time (e.g., 97%)
    #   pick the highest valued action, otherwise pick a random action
    #   (this is one way to trade off exploration exploitation)

    # *** CODE TO REMOVE *** - start
    # The code below allows a user to interactively select the action, this
    #   code is provided only so you can understand the problem (and maybe
    #   play with it), you should cut this code out once you implement
    #   your learning system.
    # print("The maze:")
    # show_maze(game, maze, player, opponent)
    # print()
    # while True:
    #     s = f"Action ({game.MIN_ACTION}-{game.MAX_ACTION}): "
    #     the_action = int(input(s))
    #     print(f"Action is {the_action}")
    #     if (the_action >= game.MIN_ACTION) and (the_action <= game.MAX_ACTION):
    #         break
    # *** CODE TO REMOVE *** - end

    print("The maze:")
    show_maze(game, maze, player, opponent)
    print()
    while True:
        s = f"Action ({game.MIN_ACTION}-{game.MAX_ACTION}): "
        the_action = int(0)
        print(f"Action is {the_action}")
        if (the_action >= game.MIN_ACTION) and (the_action <= game.MAX_ACTION):
            break

    return the_action


def test_pick_action(game, maze, player, opponent):
    # Add code here to construct a state number from the state information
    #   (player_point and opponent_point)
    # Use your state to pick an action by looking up the value of actions
    #   for your state, and picking the highest valued action (if more than
    #   one action has the highest value pick amongst those actions
    #   randomly)

    # *** CODE TO REMOVE *** - start
    # The code below allows a user to interactively select the action, this
    #   code is provided only so you can understand the problem (and maybe
    #   play with it), you should cut this code out once you implement
    #   your learning system.
    # print("The maze:")
    # show_maze(game, maze, player, opponent)
    # print()
    # while True:
    #     s = f"Action ({game.MIN_ACTION}-{game.MAX_ACTION}): "
    #     the_action = int(input(s))
    #     print(f"Action is {the_action}")
    #     if (the_action >= game.MIN_ACTION) and (the_action <= game.MAX_ACTION):
    #         break
    # *** CODE TO REMOVE *** - end

    print("The maze:")
    show_maze(game, maze, player, opponent)
    print()
    while True:
        s = f"Action ({game.MIN_ACTION}-{game.MAX_ACTION}): "
        the_action = int(0)
        print(f"Action is {the_action}")
        if (the_action >= game.MIN_ACTION) and (the_action <= game.MAX_ACTION):
            break

    return the_action


def train_learner(game, action, reward):
    # Update your learner model with the state (player_point,opponent_point),
    #   action, and reward action provided, note that you will likely have
    #   to keep track of the previous state and action to make this work
    return


def main(num_train_games, num_test_games):
    game = Game()
    player = Point(2, 2)
    opponent = Point(2, 2)
    maze = setupMaze()

    for i in range(1, num_train_games + 1):
        print(f"Training game {i}")
        initialize(game, maze, player, opponent)
        game_reward = 0.0
        while True:
            action = train_pick_action(game, maze, player, opponent)
            reward = execute_action(game, maze, player, opponent, action)
            game_reward = game_reward + reward
            train_learner(game, action, reward)
            if reward != game.REWARD_OTHER:
                break

        print(f"Reward for training game {i} is {game_reward}")

    total_reward = 0
    for i in range(1, num_test_games + 1):
        print(f"Test game {i}")
        initialize(game, maze, player, opponent)
        game_reward = 0.0
        while True:
            action = test_pick_action(game, maze, player, opponent)
            reward = execute_action(game, maze, player, opponent, action)
            game_reward = game_reward + reward
            if reward != game.REWARD_OTHER:
                break
        print(f"Reward for test game {i} is {game_reward}")
        total_reward = total_reward + game_reward
    print(f"Average test reward = {(total_reward / num_test_games)}")


if len(sys.argv) != 3:
    print("Usage: python3 SimpleMaze.py <numtraingames> <numtestgames>")
else:
    main(int(sys.argv[1]), int(sys.argv[2]))