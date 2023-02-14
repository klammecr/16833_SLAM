'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit   = 1
        self._z_short = 0.1
        self._z_max   = 0.1
        self._z_rand  = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2
        
        # Store oocupancy map
        self.map = occupancy_map

        # Store rays
        self.rays = self.map._occupancy_map.copy()
        
    def sensor_location(self, x):
        """Find sensor's position based on state of robot

        Args:
            x (list): state of robot represented by a particle
        """
        #get x, y, theta of robot from state
        rx, ry, theta = x
        
        #distance of laser in front of robot
        r = 25
        
        #compute laser's location
        lx = rx - r*np.cos(theta) 
        ly = ry + r*np.sin(theta)
        
        return [lx, ly, theta]
    
    def ray_casting(self, x, angle):
        """ray casting algorithm to find true range 

        Args:
            x (list): state of the range sensor represented by a particle
            angle (int): angle of laser beam
        """
        #unpack particle
        rx, ry, rtheta = x
        
        #measure angle of laser beam in global reference frame
        theta = rtheta + angle
        
        #perform ray casting
        #initialize positions to positions of range sensor
        cx, cy = rx, ry
        
        #iterate till wall is hit or the ray reaaches edge of map
        while True:
            #if ray goes beyond edge
            if cx < 0 or cx > self.map._size_x - 1 or \
                cy < 0 or cy > self.map._size_y - 1:
                    break
            
            #if ray hits the wall
            if self.map._occupancy_map[int(cy)][int(cx)] != 0:
                break
            
            #update x and y coordinates
            cx = cx - np.cos(theta)
            cy = cy + np.sin(theta)
            
            #self.rays[int(cy)][int(cx)] = -2
            #print("ray = ", cx, cy)
            
        z_gt = math.sqrt((cx-rx)**2 + (cy-ry)**2)    
        return z_gt
    
    def get_true_ranges(self, x):
        """function to find true range for a given state(x,y,theta) of robot

        Args:
            x (list): state of range sensor
        """
        #array to store true ranges
        z_gt = []
        
        #iterate based on subsampling
        for i in range(1, 181, self._subsampling):
            #compute range using ray casting
            z_unscaled = self.ray_casting(x, i*(math.pi/180))
            
            #scale range using map resolution
            z_scaled = z_unscaled * self.map._resolution
            
            #add range to list
            z_gt.append(z_scaled)
        
        return z_gt        
    
    def sensor_probs(self, z_t, z_gt):
        """function to compute probabiltiies of measurement

        Args:
            z_t (list): measured data
            z_gt (list): ground truth data
        """
        #initialize probabilities
        p1, p2, p3, p4 = [], [], [], []
        
        #hit probability
        for (p, q) in zip(z_t, z_gt):
            prob = 0
            if z_t <= self._max_range:
                #compute normalization factor
                n = 1/norm.cdf(self._max_range, loc=q, scale=self._sigma_hit)
                #compute probability
                prob = n*math.exp(-0.5*((p-q)/self._sigma_hit)**2)
            
            p1.append(prob)
            
        #short probability
        for (p, q) in zip(z_t, z_gt):
            prob = 0
            if p <= q:
                n = 1/(1 - math.exp(-self._lambda_short*q))
                prob =  n*self._lambda_short*math.exp(-self._lambda_short*p)
        p2.append(prob)
        
        #max probability
        for (p, q) in zip(z_t, z_gt):
            prob = 0
            if p > self._max_range:
                prob = 1
                 
            p3.append(prob)
        
        #rand probability
        for (p, q) in zip(z_t, z_gt):
            prob = 0
            if p < self._max_range:
                prob = 1/(self._max_range)
                
            p4.append(prob)
        
        return p1, p2, p3, p4
        

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 1.0
        
        #find laser position based on robot position
        x_sensor = self.sensor_location(x_t1)
        
        #perform ray casting to find GT ranges at various angles
        z_gt = self.get_true_ranges(x_sensor)
        
        #find probabilities
        p1, p2, p3, p4 = self.sensor_probs(z_t1_arr, z_gt)
        
        #aggregate probabilities
        p = self._z_hit*p1 + \
            self._z_short*p2 + \
            self._z_max*p3 + \
            self._z_rand*p4
        
        #multiply probabilities
        prob_zt1 = np.prod(p)
        
        return prob_zt1


if __name__ == "__main__":
    src_path_map = './../data/map/wean.dat'
    map1 = MapReader(src_path_map)

    sm = SensorModel(map1)
    
    z_gt = sm.get_true_ranges([590, 145, 0])
    print(z_gt)
    
    #plt.imshow(sm.map._occupancy_map, 'gray')
    plt.imshow(sm.rays, 'gray')
    plt.savefig('./rays.png')