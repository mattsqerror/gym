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
#from gym.envs.classic_control import rendering

class OpenMazeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, walls=None):
        
        
        self.init_state = np.array([.1,9.9])        
        
        #self.step_size = 0.1
        self.step_size = 0.5
        
        self.min_x = 0.
        self.max_x = 10.
        self.min_y = 0.
        self.max_y = 10.
        
        self.goal_x = self.max_x - 0.5
        self.goal_y = self.min_y
        
        self.goal_state = np.array([self.goal_x, self.goal_y])

        self.low = np.array([self.min_x, self.min_y])
        self.high = np.array([self.max_x, self.max_y])

        self.viewer = None

        self.action_space = spaces.Box(low=-np.pi,high=np.pi,shape=(1,))
        self.observation_space = spaces.Box(self.low, self.high)
        
        self.intersect_flag = False

        self._seed()
        self.reset()
        
    def _intersect(self,xy1,xy2):
        xy1 = xy1.astype(float)
        xy2 = xy2.astype(float)
        
        if len(xy1.shape) == 1:
            xy1 = xy1[None,:]
        if len(xy2.shape) == 1:
            xy2 = xy2[None,:]
        
        n_rows_1,n_cols_1 = xy1.shape        
        n_rows_2,n_cols_2 = xy2.shape
        
        x1 = np.tile(xy1[:,[0]],(1,n_rows_2))
        x2 = np.tile(xy1[:,[2]],(1,n_rows_2))
        y1 = np.tile(xy1[:,[1]],(1,n_rows_2))
        y2 = np.tile(xy1[:,[3]],(1,n_rows_2))
        
        xy2 = xy2.T
        
        x3 = np.tile(xy2[[0],:],(n_rows_1,1))
        x4 = np.tile(xy2[[2],:],(n_rows_1,1))
        y3 = np.tile(xy2[[1],:],(n_rows_1,1))
        y4 = np.tile(xy2[[3],:],(n_rows_1,1))
        
        x4_x3 = x4-x3
        y1_y3 = y1-y3
        y4_y3 = y4-y3
        x1_x3 = x1-x3
        x2_x1 = x2-x1
        y2_y1 = y2-y1
        
        numerator_a = x4_x3 * y1_y3 - y4_y3 * x1_x3
        numerator_b = x2_x1 * y1_y3 - y2_y1 * x1_x3
        denominator = y4_y3 * x2_x1 - x4_x3 * y2_y1
        
        u_a = numerator_a / denominator
        u_b = numerator_b / denominator
 
        int_b = np.logical_and(np.logical_and(np.logical_and((u_a >= 0), (u_a <= 1)), (u_b >= 0) ),(u_b <= 1)) 
        
        return int_b
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # action = np.sign((self.state[0]+math.pi/2) * self.state[1])+1
        r = np.random.randn(1) * (np.pi / 3 ) / 2
        
        action_vec = np.array([math.cos(action), math.sin(action)])
                
        next_mat = np.array([[math.cos(r), -math.sin(r)],[math.sin(r), math.cos(r)]])
        
        if self.state[0] > 5 and self.state[0] < 7 and self.state[1] > 5 and self.state[1] < 7:        
            next_state = -np.dot(next_mat, action_vec) * self.step_size + self.state
            #print self.state
        else:
            next_state = np.dot(next_mat, action_vec) * self.step_size + self.state
        
        if (next_state[0] < self.min_x) or \
           (next_state[1] < self.min_y) or \
           (next_state[0] > self.max_x) or \
           (next_state[1] > self.max_y):
            self.intersect_flag = True
        else:
            self.state = next_state
            self.intersect_flag = False
        
        
        
        #done = bool(np.linalg.norm(self.state - self.goal_state) < 0.5)
        done = False
        reward = -1.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.init_state
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_x - self.min_x
        scale_x = screen_width/world_width
        
        world_height = self.max_y - self.min_y
        scale_y = screen_height/world_height
        
        cartwidth=10
        cartheight=10

        ex_size=0.05

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            ## agent            
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            self.cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            self.cart.add_attr(self.carttrans)
            
            self.cart.set_color(0,0,0)
            self.viewer.add_geom(self.cart)
            
            ## outer boundary
            l,r,t,b = self.min_x*scale_x,self.max_x*scale_x,self.max_y*scale_y,self.min_y*scale_y
            boundary = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)], False)
            self.viewer.add_geom(boundary)
            
        
        x = self.state[0]
        y = self.state[1]        
        
        
#        ex1 = rendering.Line(((x-ex_size)*scale_x,(y+ex_size)*scale_y),((x+ex_size)*scale_x,(y-ex_size)*scale_y))        
#        ex2 = rendering.Line(((x-ex_size)*scale_x,(y-ex_size)*scale_y),((x+ex_size)*scale_x,(y+ex_size)*scale_y))        
#
#        self.viewer.add_geom(ex1)
#        self.viewer.add_geom(ex2)

        if self.intersect_flag:        
            self.cart.set_color(1,.2,.2) # red
        else:
            self.cart.set_color(0,0,0) # black
        

        cartx = (x-self.min_x)*scale_x
        carty = (y-self.min_y)*scale_y
        
        
        
        self.carttrans.set_translation(cartx, carty)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
