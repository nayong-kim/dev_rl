# trainer.py
  
  
import os
import time
import numpy as np
import random
from game import Game
from game import NUM_ACTIONS, NUM_CHANNELS
from agent import Agent
import pickle
import time
from datetime import datetime
  
class Trainer :
    
    def __init__(self, field_width, field_height,
                 episodes=500,
                 initial_epsilon=1.,
                 min_epsilon=0.1,
                 exploration_ratio=0.5,
                 max_steps=100,
                 batch_size=64,
                 min_replay_memory_size=100,
                 replay_memory_size=5000,
                 target_update_freq=5,
                 ):
        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.exploration_ratio = exploration_ratio
        self.enable_save = True
        self.save_freq = 10
        self.save_dir = 'checkpoint'+ datetime.today().strftime("%Y%m%d%H%M%S")
        self.max_average_length = 0
        
        
        self.env = Game(field_width = field_width, 
                        field_height = field_height)
        self.agent = Agent(field_size = (field_width, field_height),
                           gamma = 0.99,
                           batch_size =batch_size,
                           min_replay_memory_size = min_replay_memory_size, 
                           replay_memory_size = replay_memory_size, 
                           target_update_freq = target_update_freq
                           )
        self.epsilon_decay = (initial_epsilon-min_epsilon)/(exploration_ratio*episodes)

        self.current_episode = 0
        
        
    def preview(self, render_fps, disable_exploration=False, save_dir=None):
            if save_dir is not None and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # if save_dir is not None:
            #     self.env.save_image(save_path=save_dir+'/0.png')

            current_state = self.env.reset()
            
            Done = False
            
            steps = 0
            self.epsilon = 0.0001
            while not Done and steps < self.max_steps:
                if disable_exploration or random.random() > self.epsilon:
                    
                    action = np.argmax(self.agent.get_q_values(np.array([current_state])))
                else:
                    action = np.random.randint(NUM_ACTIONS)
                action = random.randint(0, NUM_ACTIONS)
                print(action)
                field, next_state, reward, Done = self.env.step(action)
                current_state = next_state
                steps += 1
                # time.sleep(0.001)
                # print(f"field = \n{field}")
                print(f"{steps}/{self.max_steps}")
                # print(f"pos = \n{next_state}")
                self.env.render(fps=render_fps, pos = next_state)
                # if save_dir is not None:
                #     self.env.save_image(save_path=save_dir+'/{}.png'.format(steps))


            
    def train(self):

        while self.current_episode < self.episodes:
            
            current_state = self.env.reset()
            current_state = current_state.reshape(-1, self.env.screen_width, self.env.screen_height, NUM_CHANNELS)
                
            done = False
            
            steps = 0
            self.epsilon = 0.5
            while not done and steps < self.max_steps:
                start = time.time()
                if random.random() > self.epsilon:
                    # print(current_state)
                    # print(f"current_state.ndim:{current_state.ndim}, shape={current_state.shape}")
                    action = np.argmax(self.agent.get_q_values(current_state))
                else:
                    action = np.random.randint(NUM_ACTIONS - 1)
                action = random.randint(0, NUM_ACTIONS - 1)
                # print(action)
                next_state, next_pos, reward, done = self.env.step(action)
                next_state = next_state.reshape(-1, self.env.screen_width, self.env.screen_height, NUM_CHANNELS)
                self.agent.update_replay_memory(current_state, action, reward, next_state, done)
                current_state = next_state
                
               r4
                
                
                steps += 1
                
                # time.sleep(0.001)
                # print(f"field = \n{field}")
                end = time.time()
                if steps % 10 ==0:
                    print(f"{end - start:.5f}sec, steps:{steps:5d}/{self.max_steps:5d}, \
                          episode:{self.current_episode:5d}/{self.episodes:5d}")
                # print(f"pos = \n{next_state}")
                self.env.render(fps=60, pos = next_pos)
                # if save_dir is not None:
                #     self.env.save_image(save_path=save_dir+'/{}.png'.format(steps))
            
            self.agent.increase_target_update_counter()
            loss = self.agent.train()
            self.current_episode += 1
            
              # save model, training info
            if self.enable_save and self.current_episode % self.save_freq == 0:
                self.save(str(self.current_episode))


            
    def quit(self):
        self.env.quit()
        
    def save(self, suffix):
        self.agent.save(
            self.save_dir+'/model_{}.h5'.format(suffix),
            self.save_dir+'/target_model_{}.h5'.format(suffix)
        )

        dic = {
            'replay_memory': self.agent.replay_memory,
            'target_update_counter': self.agent.target_update_counter,
            'current_episode': self.current_episode,
            'epsilon': self.epsilon,
            # 'summary': self.summary,
            'max_average_length': self.max_average_length
        }

        with open(self.save_dir+'/training_info_{}.pkl'.format(suffix), 'wb') as fout:
            pickle.dump(dic, fout)    