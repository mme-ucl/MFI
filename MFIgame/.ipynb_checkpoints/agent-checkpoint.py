from turtle import update
import torch
import random
import numpy as np
from collections import deque
from MFIgame import MFIgame
from model import Linear_QNet, QTrainer
import time
import matplotlib.pyplot as plt
from helper import plot, plot_extended

fig = plt.gcf()
fig.set_size_inches(20,18) 

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


name_list = [ "Simulations Steps", "Initial Position", "MetaD Width", "MetaD Height", "MetaD Biasfactor"]
param_extream = [[25000, 150000], [-1.75, 1.75], [0.05, 0.15], [0.5, 3], [1, 16]]

class Agent:

    def __init__(self):
        self.n_games = 0
        self.n_simulations = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.game_memory = []
        self.model = Linear_QNet(2, 256, 5)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        #variables for plot
        #game var
        self.idx_new_simulation = []
        self.plot_aad = []
        self.plot_mean_aad = []
        self.plot_Aofe = []
        self.plot_mean_Aofe = []
        self.total_aad = 0
        self.total_Aofe = 0
        #total var
        self.plot_aad_game = []
        self.plot_mean_aad_game = []
        self.plot_Aofe_game = []
        self.plot_mean_Aofe_game = []
        self.total_aad_game = 0
        self.total_Aofe_game = 0
        #record var
        self.record = 2.5
        self.record_Aofe = np.float64(100.0)
        #input var
        self.nsteps_list = []        
        self.ipos_list = []        
        self.sigma_list = []
        self.height_list = []
        self.bf_list = []
        
    def update_progress(self, AAD, Aofe, x, y, Ftot_den, FES, ofe, final_move, record):
        #record AAD of FES
        self.plot_aad.append(AAD)
        self.total_aad += AAD
        self.plot_mean_aad.append(self.total_aad / self.n_simulations)
        
        #record ofe of force
        self.plot_Aofe.append(Aofe)
        self.total_Aofe += Aofe
        self.plot_mean_Aofe.append(self.total_Aofe / self.n_simulations)
        
        self.nsteps_list.append(final_move[0])
        self.ipos_list.append(final_move[1])
        self.sigma_list.append(final_move[2])
        self.height_list.append(final_move[3])
        self.bf_list.append(final_move[4])
        
        #record AAD and ofe of game end
        game_over = 0
        if len(self.plot_aad_game) < self.n_games:
            #record AAD of FES
            self.plot_aad_game.append(AAD)
            self.total_aad_game += AAD
            self.plot_mean_aad_game.append(self.total_aad_game / self.n_games)
            
            #record ofe of force
            self.plot_Aofe_game.append(Aofe)
            self.total_Aofe_game += Aofe
            self.plot_mean_Aofe_game.append(self.total_Aofe_game / self.n_games)
            
            game_over = 1
            self.idx_new_simulation.append(len(self.plot_aad)-1)
        
        #update plot
        plot_extended(self.plot_aad, self.plot_mean_aad, self.plot_Aofe, self.plot_mean_Aofe, self.plot_aad_game, self.plot_mean_aad_game, self.plot_Aofe_game, self.plot_mean_Aofe_game, x, y, Ftot_den, FES, ofe, [self.nsteps_list, self.ipos_list, self.sigma_list, self.height_list, self.bf_list], name_list, param_extream, record, game_over, self.idx_new_simulation)
        
    
    
    #state of game
    # def get_state(self, game):
    # should be AAD and AAD_f
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
    def remember_game(self, state, action, reward, next_state, done):
        self.game_memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
    
    

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # print("\n\n### Actions: size, type =", len(actions), type(actions))#, "\n\n")
        # print("### states: size, type =", len(states), type(states))
        # print("### rewards: size, type =", len(rewards), type(rewards))
        # print("### next states: size, type =", len(next_states), type(next_states))
        # print("### next does: size, type =", len(dones), type(dones))
        # print("ACTIONS\n", actions, "\n\n")
        # print("STATES\n",states, "\n\n")
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_game_memory(self):
        mini_sample = self.game_memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        if rewards[-1] <= self.record:
            reward = 0
            self.record = rewards[-1]
        else:
            reward = (rewards[-1] - self.record) * 10        
        
        self.trainer.train_step(states, actions, reward, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        self.epsilon = 25 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 50) < self.epsilon:
            length_rand = int(25000 * (1 + random.randint(0, 5)))  # 25000 * (1 + [0 -- 11]) => [25_000, 150_000]
            ipos_rand = random.randint(25, 375) / 100 - 2 # [-1.75, 1.75]
            width_rand = 0.05 + 0.0001*random.randint(0, 1000)  # [0.05, 0.15]
            height_rand = 0.5 + 0.0025*random.randint(0, 1000)  # [0.5, 3.0]
            bf_rand = 1 + 0.015*random.randint(0, 1000) # [1, 16]
            
            
            prediction = [length_rand, ipos_rand, width_rand, height_rand, bf_rand]
            print("***random move --> n_steps:" ,prediction[0], " | ipos:" ,round(prediction[1],2), " | sigma:" ,round(prediction[2],4), " |  height:" ,round(prediction[3],3), " | biasfactor:" ,round(prediction[4],2)) 
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            pred = prediction.tolist()
            #predictions = [n_steps, ipos, metad_width, metad_height, biasfactor]
            #predictions_range = [[25_000, 150_000], [-1.75, 1.75], [0.05, 0.15], [0.5, 3], [1, 16]]
            prediction = [25000 * int(1 + pred[0]*5.99), pred[1]*3.5 - 1.75, 0.05 + pred[2]*0.1, 0.5 + pred[3]*2.5, 1 + pred[4]*15]
            print("***Agent move --> n_steps:" ,prediction[0], " | ipos:" ,round(prediction[1],2), " | sigma:" ,round(prediction[2],4), " |  height:" ,round(prediction[3],3), " | bf:" ,round(prediction[4],2)) 
            # move = torch.prediction #.item()
        
        return prediction


def train():
        
    done = 1    #activated when time is up (and at beginning).
    
    agent = Agent()
    game = MFIgame()
    
    
    for sim_itteration in range(100):
        
        #Activated at beginning of every game
        if done == 1:
            done = 0
            game.reset()
            agent.game_memory = []
            print("\n~~~~~~~~~~STARTING NEW GAME", agent.n_games, "~~~~~~~~~~\n")
            start = time.time()
            [x, y, Ftot_den, FES, AD, AAD, ofe, Aofe] = game.MFI_new()
            agent.n_simulations += 1
            agent.update_progress(AAD, Aofe, x, y, Ftot_den, FES, ofe, [25000, 0.0, 0.1, 1, 10], agent.record)

        
        
        print("\nNew simulation:",  sim_itteration, end="")     
        # get old state
        state_old = [int(AAD*100), int(Aofe*10)]

        # get move
        final_move = agent.get_action(state_old)
        

        # perform move
        [x, y, Ftot_den, FES, AD, AAD, ofe, Aofe] = game.MFI_action(final_move)
        agent.n_simulations += 1
                
        #new state
        state_new = [int(AAD*100), int(Aofe*10)]
        
        #Get reward from FES
        if AAD < agent.record:
            agent.record = AAD
        #     agent.model.save()
        #     reward = 10
        # if AAD > (agent.record * 1.2):
        #     reward = -20
        # elif AAD > agent.record:
        #     reward = -10
        
        reward = AAD
            
            
        # #Get reward from ofe
        # #ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR 
        # print("\n\n\n Aofe:", Aofe, " ||  record_Aofe:  ", record_Aofe)
        # print("Aofe:", type(Aofe), " ||  record_Aofe:  ", type(record_Aofe))
        # print("Aofe:", np.shape(Aofe), " ||  record_Aofe:  ", np.shape(record_Aofe))        
        # print("\n\nhere it comes: Aofe > record_Aofe * 1.2 =>     ", Aofe > (record_Aofe * 1.2))
        # #ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR 
        # if Aofe < record_Aofe:
        #     record_Aofe = ofe
        #     # agent.model.save()
        #     reward += 10
        # #ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR 
        # if Aofe > record_Aofe * 1.2:
        #     reward += -20
        # elif Aofe > record:
        #     reward += -10            
        
        
        print('Simulation', sim_itteration+1, ': AAD =', AAD, '| OFE =', Aofe)#, '| Reward =', reward)
            
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember_game(state_old, final_move, reward, state_new, done)

        #check if time is up -> Game over
        if time.time() - start > 20:
            done = 1
                     
        #if time is up -> Game over. Train long memory, plot progress and restart 
        if done == 1:           
            agent.train_game_memory()
            print('Game', agent.n_games, 'finished --> AAD', round(AAD,2), ' | Aofe:', round(Aofe,2),  ' | time:', round(time.time() - start,2))#, ' | reward:', int(reward)
            agent.n_games += 1         
        
        agent.update_progress(AAD, Aofe, x, y, Ftot_den, FES, ofe, final_move, agent.record)
   
    
    
if __name__ == '__main__':
    train()
    