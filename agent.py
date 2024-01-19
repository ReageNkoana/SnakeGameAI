import torch
import random
import numpy as np
from collections import deque
from snakegameAI import SnakeGameAI, Direction, Point  # importing the class SnakeGameAI from the file snakegameAI.py
from model import Linear_Qnet, Qtrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate...must be smaller than one if you play with it
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_Qnet(11, 256, 3)
        self.trainer = Qtrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: model trainer



    def get_state(self, snakegameAI):
        head = snakegameAI.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snakegameAI.direction == Direction.LEFT
        dir_r = snakegameAI.direction == Direction.RIGHT
        dir_u = snakegameAI.direction == Direction.UP
        dir_d = snakegameAI.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snakegameAI.is_collision(point_r)) or
            (dir_l and snakegameAI.is_collision(point_l)) or
            (dir_u and snakegameAI.is_collision(point_u)) or
            (dir_d and snakegameAI.is_collision(point_d)),

            # Danger right
            (dir_u and snakegameAI.is_collision(point_r)) or
            (dir_d and snakegameAI.is_collision(point_l)) or
            (dir_l and snakegameAI.is_collision(point_u)) or
            (dir_r and snakegameAI.is_collision(point_d)),

            # Danger left
            (dir_d and snakegameAI.is_collision(point_r)) or
            (dir_u and snakegameAI.is_collision(point_l)) or
            (dir_r and snakegameAI.is_collision(point_u)) or
            (dir_l and snakegameAI.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            snakegameAI.food.x < snakegameAI.head.x,  # food left
            snakegameAI.food.x > snakegameAI.head.x,  # food right
            snakegameAI.food.y < snakegameAI.head.y,  # food up
            snakegameAI.food.y > snakegameAI.head.y,  # food down
            ]
        return np.array(state, dtype=int)





    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached




    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # returns a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)





    def get_action(self, state):
        # random moves : tradeoff exploration/  exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



def train():
    plot_sores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snakegameAI = SnakeGameAI()
    while True:

        # get old state
        state_old = agent.get_state(snakegameAI)

        # get move
        final_move = agent.get_action(state_old)

        # perform move move and get new state
        reward, done, score = snakegameAI.play_step(final_move)
        state_new = agent.get_state(snakegameAI)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            snakegameAI.reset()
            agent.n_games +=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record: ', record)

            plot_sores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_sores, plot_mean_scores)





if __name__ == '__main__':
    train()




































