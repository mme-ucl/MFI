import torch
import random
import numpy as np
from collections import deque
from MFIgame import MFIgame
from model import Linear_QNet, QTrainer
import time
import matplotlib.pyplot as plt
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(2, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    #state of game
    # def get_state(self, game):
    # should be AAD and AAD_f
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            width_rand = 0.08 + 0.0001*random.randint(0, 400)
            height_rand = 0.01*random.randint(0, 1000)
            bf_rand = 0.01*random.randint(500, 2000)
            length_rand = random.randrange(1000, 300000,1000)
            
            
            prediction = [width_rand, height_rand, bf_rand, length_rand]
            print("*************random move***************** --> predictions" ,prediction) 
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            # move = torch.prediction #.item()
        
        return prediction


def train():
    plot_aad = []
    plot_mean_aad = []
    plot_aad_f = []
    plot_mean_aad_f = []
    total_aad = 0
    total_aad_f = 0
    record = 10
    record_f = 0
    itteration = 0
    
    done = False
    
    agent = Agent()
    game = MFIgame()
    
    print("starting new game \n\n")
    start = time.time()
    AAD, AAD_f = game.MFI_new()
    
    for sim_itteration in range(100):
        
        print("starting new simulation:",  sim_itteration, "\n")     
        # get old state
        state_old = np.array([int(AAD*100), int(AAD_f*100)], dtype=int)

        # get move
        final_move = agent.get_action(state_old)
        
        print("\n type final move:" , type(final_move), "\nfinal move", final_move)

        # perform move
        AAD, AAD_f = game.MFI_action(final_move)
                
        #new state
        state_new = int(AAD*100), int(AAD_f*100)

        #Get reward
        if AAD < record:
            record = AAD
            # agent.model.save()
            reward = 10
        if AAD > record * 1.2:
            reward = -10
        if min(final_move) < 0:
                reward = -20

        
        print('Game', agent.n_games, 'Score', AAD, 'Record:', record)
            


        # train memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if time.time() - start < 60:
            done = True
            
        
        if done == True:
            
            done == False
            agent.n_games += 1
            
            # train long memory, plot result
            agent.train_long_memory()

            print('Game', agent.n_games, 'Score', AAD, 'Record:', record)

            #record AAD of FES
            plot_aad.append(AAD)
            total_aad += AAD
            mean_aad = total_aad / agent.n_games
            plot_mean_aad.append(mean_aad)
            
            #record AAD of force
            plot_aad_f.append(AAD_f)
            total_aad_f += AAD_f
            mean_aad_f = total_aad_f / agent.n_games
            plot_mean_aad_f.append(mean_aad_f)

            plot(plot_aad, plot_mean_aad)    
            
            #restart clock
            start = time.time()
            
            #restart game
            print(">>>>>starting game", agent.n_games + 1,  "\n\n")
            start = time.time()
            AAD, AAD_f = game.MFI_new()
    
    
if __name__ == '__main__':
    train()
    