# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:54:35 2016

@author: matt
"""
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import h5py
import scipy
#from gym.envs.classic_control import rendering

class CoilEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, h5fn):
        
        self.h5fn = h5fn
        
        g = h5py.File(self.h5fn,'r')
        test_inds = g['test_inds'][:]
        self.images = g['images'][:][test_inds].transpose(0,2,3,1).reshape(100,68,32,32,3)
        self.images_enc = g['images_enc'][:][test_inds].reshape(100,68,-1)
        g.close()
        
        self.init_angle_ind = np.random.choice(68) # 72 angles per object minus 4 "training angles"
        self.init_object_ind = np.random.choice(100) # 100 objects
        self.init_state = self.images_enc[self.init_object_ind,self.init_angle_ind]      
        
        self.viewer = None
        

        self.low = np.min(self.images_enc.reshape(6800,-1), axis=0)
        self.high = np.max(self.images_enc.reshape(6800,-1), axis=0)
        
        
        self.action_space = spaces.Discrete(68) # 72 angles per object minus 4 "training angles"
        self.observation_space = spaces.Box(self.low, self.high)
        
        self.object_ind = self.init_object_ind
        self.angle_ind = self.init_angle_ind

        self._seed()
        self.reset()
        

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        reward = 0  # no rewards for environment
        done = False # never done
        
        if action == 0:
            self._reset()
        else:
            self.angle_ind = (self.angle_ind + action) % 68
            self.state = self.images_enc[self.object_ind,self.angle_ind]

        return np.array(self.state), reward, done, {'object_ind': self.object_ind, 'angle_ind': self.angle_ind}

    def _reset(self, obj=None, ang=None):
        if ang is None:
            self.angle_ind = np.random.choice(68) # 72 angles per object minus 4 "training angles"
        else:
            self.angle_ind = ang
        if obj is None:
            self.object_ind = np.random.choice(100) # 100 objects
        else:
            self.object_ind = obj
        self.state = self.images_enc[self.object_ind,self.angle_ind]      
 
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        #screen_width = 600
        #screen_height = 400


        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
            
        img = scipy.misc.imresize(self.images[self.object_ind,self.angle_ind],(256,256))
        self.viewer.imshow(img)

        #return  
