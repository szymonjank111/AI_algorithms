import random
import matplotlib.pyplot as plt
import pygame
import torch
import os
import time
import pickle

from food import Food
from snake import Snake, Direction


epsilons = [0.1, 0.2, 0.3]
# epsilons = [0.3]
discounts = [0.8, 0.9, 0.99]


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds)

    is_training = False
    # is_training = True

    epsilon = 0.2
    discount = 0.9
    learning_rate = 0.11

    # for epsilon in epsilons:
    #     for discount in discounts:

    agent = QLearningAgent(block_size, bounds, epsilon, discount, learning_rate, is_training)
    scores = []
    rewards = []
    my_reward = 0
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode, total_episodes = 0, 100
    while episode < total_episodes and run:
        pygame.time.delay(50)  # Adjust game speed, decrease to learn agent faster

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                    "snake_body": snake.body,  # The last element is snake's head
                    "snake_direction": snake.direction}

        direction = agent.act(game_state, reward, is_terminal)
        reward = -0.001
        is_terminal = False
        snake.turn(direction)
        snake.move()
        my_reward -= reward
        reward += snake.check_for_food(food)
        my_reward += reward
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(1)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()
            episode += 1
            reward -= 0.999
            my_reward -= 0.999
            is_terminal = True

        if is_terminal:
            rewards.append(my_reward)
            my_reward = 0
        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    # print(f"Check out mean_reward for e: {agent.eps} and d: {agent.discount}")
    print(f"Avg score: {sum(scores) / len(scores)}\n")
    """ This will create a smoothed mean score per episode plot.
    I want you to create smoothed mean reward per episode plots, that's how we evaluate RL algorithms!"""
    # scores = torch.tensor([scores], dtype=torch.float).unsqueeze(0)
    # scores = torch.nn.functional.avg_pool1d(scores, 31, stride=1)
    # plt.plot(torch.squeeze(scores))
    # plt.savefig(f"score_reward_e_{agent.eps}_d_{agent.discount}.png")
    # print("Check out score_reward.png")

    # rewards = torch.tensor([rewards], dtype=torch.float).unsqueeze(0)
    # rewards = torch.nn.functional.avg_pool1d(rewards, 31, stride=1)
    # plt.plot(torch.squeeze(rewards))
    # plt.savefig(f"mean_reward_e_{agent.eps}_d_{agent.discount}.png")
    # plt.clf()

    if is_training:
        agent.dump_qfunction()
    pygame.quit()


class QLearningAgent:
    def __init__(self, block_size, bounds, epsilon, discount, lr, is_training=True, load_qfunction_path=None):
        """ There should be an option to load already trained Q Learning function from the pickled file. You can change
        interface of this class if you want to."""
        self.block_size = block_size
        self.bounds = bounds
        self.is_training = is_training
        self.Q = torch.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        if is_training is False:
            self.load_qfunction()
        self.obs = None
        self.action = None
        self.discount = discount
        self.eps = epsilon
        self.lr = lr

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        """ Update Q-Learning function for the previous timestep based on the reward, and provide the action for the current timestep.
        Note that if snake died then it is an end of the episode and is_terminal is True. The Q-Learning update step is different."""
        new_obs = self.game_state_to_observation(game_state, self.bounds, self.block_size)
        if random.random() < self.eps:
            new_action = random.randint(0, 3)
        else:
            new_action = torch.argmax(self.Q[new_obs])

        if self.action:
            if not is_terminal:
                update = reward + self.discount * torch.max(self.Q[new_obs]) - self.Q[self.obs][self.action]
            else:
                update = reward - self.Q[self.obs][self.action]
            self.Q[self.obs][self.action] += self.lr * update

        self.action = new_action
        self.obs = new_obs
        return Direction(int(new_action))

    @staticmethod
    def game_state_to_observation(game_state, bounds, block_size):
        gs = game_state
        is_up = int(gs["food"][1] < gs["snake_body"][-1][1])
        is_right = int(gs["food"][0] > gs["snake_body"][-1][0])
        is_down = int(gs["food"][1] > gs["snake_body"][-1][1])
        is_left = int(gs["food"][0] < gs["snake_body"][-1][0])

        is_barrier_up = int(gs["snake_body"][-1][1] == 0)
        is_barrier_right = int(gs["snake_body"][-1][0] == (bounds[0] - block_size))
        is_barrier_down = int(gs["snake_body"][-1][1] == (bounds[1] - block_size))
        is_barrier_left = int(gs["snake_body"][-1][0] == 0)

        is_dir_up = int(gs["snake_direction"] == Direction.UP)
        is_dir_right = int(gs["snake_direction"] == Direction.RIGHT)
        is_dir_down = int(gs["snake_direction"] == Direction.DOWN)
        is_dir_left = int(gs["snake_direction"] == Direction.LEFT)

        return is_up, is_right, is_down, is_left, is_barrier_up, is_barrier_right, is_barrier_down, is_barrier_left, is_dir_up, is_dir_right, is_dir_down, is_dir_left

    def act_test(self, game_state: dict) -> Direction:
        new_obs = self.game_state_to_observation(game_state, self.bounds, self.block_size)
        new_action = torch.argmax(self.Q[new_obs])
        return Direction(int(new_action))

    def dump_qfunction(self):
        os.makedirs("best_agents", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"best_agents/ql_{current_time}.pickle", 'wb') as f:
            pickle.dump(self.Q, f)

    def load_qfunction(self):
        # with open(f"ql_agents/ql_2023-01-13_16:17:03.pickle", 'rb') as f:
        # with open(f"best_agents/ql_2023-01-13_18:47:31.pickle", 'rb') as f:
        with open(f"best_agents/ql_2023-01-13_16:57:09.pickle", 'rb') as f:
            qlearning_function = pickle.load(f)
            self.Q = qlearning_function


if __name__ == "__main__":
    main()
