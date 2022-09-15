import os
os.chdir("/home/antoniu/Desktop/MFI_git/MFI_master/MFI/MFIgame/MFIgame_parallel")

from turtle import update
import torch
import random
import numpy as np
from collections import deque
# from MFIgame import MFIgame
from model_parallel import Linear_QNet, QTrainer
import time
import matplotlib.pyplot as plt
from helper import plot, plot_extended

import threading
from multiprocessing import Process, Manager, Queue, Array, Lock, Pool
import concurrent.futures


import MFI1Dgame_parallel as MFI
import run_plumedgame as plumed



#analytical function
min_grid=-2
max_grid=2
nbins=201
grid = np.linspace(min_grid, max_grid, nbins)
y = 7*grid**4 - 23*grid**2
y = y - min(y)


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
        # self.reset()
        
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
     
    def update_progress_master(self, plot_progress):
        
        while len(self.plot_aad) < len(plot_progress):
            
            ii = len(self.plot_aad)
            
            self.update_progress(plot_progress[ii][0], plot_progress[ii][1], plot_progress[ii][2], plot_progress[ii][3], plot_progress[ii][4], plot_progress[ii][5], plot_progress[ii][6], plot_progress[ii][7], plot_progress[ii][8])
            #plot_progress.append([AAD, Aofe, x, y, Ftot_den, FES, ofe, final_move, AAD])  #AAD should be record
            #update_progress(self, AAD, Aofe, x, y, Ftot_den, FES, ofe, final_move, record)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
    def remember_game(self, state, action, reward, next_state, done):
        self.game_memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
        print("\n\n\nREMEMBER GAME: ")
        print((self.game_memory), "\n\n")
        print("\n\nlen GAME MEMORY:", len(self.game_memory), "\n\n")

           

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
        
        print("\n\nGAME MEMORY:", len(self.game_memory), "\n\n")

        
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

    def MFI_action(self, action, master, master_patch):
        #Run simulation
        plumed.run_langevin1D(action[0], initial_position=action[1], gaus_width=round(action[2],5), gaus_height=round(action[3],4), biasfactor=round(action[4],2))
        # os.system("rm bck.*")

        #Read the Colvar File
        position = MFI.load_position(position_name="position")

        #Read the HILLS file
        HILLS=MFI.load_HILLS(hills_name="HILLS")

        #Compute the time-independent mean force
        [X, Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)

        #Append force to master
        master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])
        
        #Patch master
        [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI.patch_FES_AD_ofe(master, X, y, nbins)
        master_patch.append([PD_patch, F_patch, OFE])
        
        return [X, y, Ftot_den, FES, AD, AAD, OFE, AOFE]
    
    
    def MFI_new(self, master, master_patch, plot_progress):
        
        #Run simulation
        plumed.run_langevin1D(25000, gaus_width=0.1, gaus_height=1, biasfactor=10)
        os.system("rm bck.*")

        #Read the Colvar File
        position = MFI.load_position(position_name="position")

        #Read the HILLS file
        HILLS=MFI.load_HILLS(hills_name="HILLS")

        #Compute the time-independent mean force
        [X, Ftot_den, Ftot_den2, Ftot, ofv_num] = MFI.MFI_1D(HILLS = HILLS, position = position, bw = 0.05)

        #Append force to master
        master.append([Ftot_den, Ftot_den2, Ftot, ofv_num])
        
        #Patch master
        [X, PD_patch, F_patch, FES, AD, AAD, OFE, AOFE] = MFI.patch_FES_AD_ofe(master, X, y, nbins)
        master_patch.append([PD_patch, F_patch, OFE])
        
        plot_progress.append([AAD, AOFE, X, y, Ftot_den, FES, OFE, [25000, 0.0, 0.1, 1.0, 10], AAD])  #AAD should be record

        
        return [X, y, Ftot_den, FES, AD, AAD, OFE, AOFE]

    # def reset(self):
    #     self.master = []
    #     self.master_patch = []


    def simulation_step(self, folder_name, state, sim_itteration, master_state, master, master_patch, plot_progress):
        
        
            done_i = 0
        
            print("\nNew simulation:",  sim_itteration, end="")     
            # get old state
            state_old = [int(state[0]*100), int(state[1]*10)]

            # get move
            final_move = self.get_action(state_old)
                        
            # perform move
            os.chdir("/home/antoniu/Desktop/MFI_git/MFI_master/MFI/MFIgame/MFIgame_parallel/" + folder_name)
            [x, y, Ftot_den, FES, AD, AAD, ofe, Aofe] = self.MFI_action(final_move, master, master_patch)
            self.n_simulations += 1
            os.chdir("/home/antoniu/Desktop/MFI_git/MFI_master/MFI/MFIgame/MFIgame_parallel/")
                    
            #new state
            state_new = [int(AAD*100), int(Aofe*10)]
            
            #Get reward from FES    
            reward = AAD
                
            # train short memory
            self.train_short_memory(state_old, final_move, reward, state_new, done_i)

            # remember
            self.remember_game(state_old, final_move, reward, state_new, done_i)
            print('Simulation', sim_itteration+1, ': AAD =', AAD, '| OFE =', Aofe)#, '| Reward =', reward)   
            
            master_state.append([AAD, Aofe])
            plot_progress.append([AAD, Aofe, x, y, Ftot_den, FES, ofe, final_move, AAD])  #AAD should be record
            
            print("\n\nlen GAME MEMORY 2:", len(self.game_memory), "\n\n")

            


def train():
    
    try: 
        os.system("rm -r folder*")
        time.sleep(0.2)
        print("old folders removed")
    except:
        print("no folders to be removed")
        
    os.mkdir("folder1")
    os.mkdir("folder2")
    os.mkdir("folder3")
    os.mkdir("folder4")
    
        
    done = 1    #activated when time is up (and at beginning).
    
    agent = Agent()
    # game = MFIgame()
    
    master_state = [[0,0], [0,0], [0,0]]
    with Manager() as manager:
        
        plot_progress = manager.list()

        p_alive = [0,0,0,0]
            
            
        for sim_itteration in range(100):
            
            #Activated at beginning of every game
            if done == 1:
                
                print("\n\n\n DONE DONE DONE DONE DONE DONE \n\n\n\n DONE DONE DONE DONE DONE DONE \n\n\nDONE DONE DONE \n\n\n")
                        
                done = 0
                # agent.reset()
                master_state = manager.list()
                master = manager.list()
                master_patch = manager.list()
                agent.game_memory = []
                print("\n~~~~~~~~~~STARTING NEW GAME", agent.n_games, "~~~~~~~~~~\n")
                start = time.time()
                [x, y, Ftot_den, FES, AD, AAD, ofe, Aofe] = agent.MFI_new(master, master_patch, plot_progress)
                master_state.append([AAD, Aofe])
                agent.n_simulations += 1
                agent.update_progress(AAD, Aofe, x, y, Ftot_den, FES, ofe, [25000, 0.0, 0.1, 1, 10], agent.record)

            #check if process inactive. if all active, loop until one becomes inactive
            if sum(p_alive) == 4:
                while sum(p_alive) == 4:
                    if p1.is_alive() == False:
                        p_alive[0] = 0
                        print("p1 has finished")
                    elif p2.is_alive() == False:
                        p_alive[1] = 0
                        print("p2 has finished")
                    elif p3.is_alive() == False:
                        p_alive[2] = 0
                        print("p3 has finished")
                    elif p4.is_alive() == False:
                        p_alive[3] = 0
                        print("p4 has finished")
                    else:
                        time.sleep(0.1)
                        
            
            if p_alive[0] == 0:  #make explorer, start from global max, long
                new_input = ("folder1", master_state[-1], sim_itteration, master_state, master, master_patch, plot_progress)
                p1 = Process(target=agent.simulation_step, args=(new_input))
                p1.start()
                p_alive[0] = 1

            elif p_alive[1] == 0:  #make explorer, start from random position
                new_input = ("folder2", master_state[-1], sim_itteration, master_state, master, master_patch, plot_progress)
                p2 = Process(target=agent.simulation_step, args=(new_input))
                p2.start()
                p_alive[1] = 1
                
            elif p_alive[2] == 0:  #make stay, start from global max
                new_input = ("folder3", master_state[-1], sim_itteration, master_state, master, master_patch, plot_progress)
                p3 = Process(target=agent.simulation_step, args=(new_input))
                p3.start()
                p_alive[2] = 1
                
            elif p_alive[3] == 0:  #make stay, start from variance max, short
                new_input = ("folder4", master_state[-1], sim_itteration, master_state, master, master_patch, plot_progress)
                p4 = Process(target=agent.simulation_step, args=(new_input))
                p4.start()
                p_alive[3] = 1
            
            print("\n\nlen GAME MEMORY 3:", len(agent.game_memory), "\n\n")



            #check if time is up -> Game over
            if time.time() - start > 10:
                done = 1
                
                p1.join()
                p2.join()
                p3.join()
                p4.join()
                p_alive = [0,0,0,0]

                        
            #if time is up -> Game over. Train long memory, plot progress and restart 
            if done == 1:           
                agent.train_game_memory()
                print('Game', agent.n_games, 'finished --> AAD', round(AAD,2), ' | Aofe:', round(Aofe,2),  ' | time:', round(time.time() - start,2))#, ' | reward:', int(reward)
                agent.n_games += 1         
            
            # agent.update_progress(AAD, Aofe, x, y, Ftot_den, FES, ofe, final_move, agent.record)
            #function to update progress
            agent.update_progress_master(plot_progress)
    
    
if __name__ == '__main__':
    train()
    