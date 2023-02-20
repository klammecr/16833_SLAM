'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from multiprocessing import Pool, Process, Queue
import numpy as np
import math
import time
import queue
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader
import ray_casting as rc

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
        self._z_hit   = 1.0
        self._z_short = 0.1
        self._z_max   = 0.1
        self._z_rand  = 100.0

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000
        
        #variable to ensure non-zero probabilities
        self.eps = 1e-5
        
        #initialize ray casting object
        self.ray_casting = rc.RayCasting(occupancy_map)
    
    def sensor_probs(self, z_t, z_gt):
        """function to compute probabiltiies of measurement

        Args:
            z_t (list): measured data
            z_gt (list): ground truth data
        """
        
        #convert list to numpy array
        z_t = np.array(z_t)
        z_gt = np.array(z_gt)
        
        #initialize probabilities
        p1, p2, p3, p4 = [], [], [], []
        
        #HIT PROBABILITY
        #initialize probabilties
        p1 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t <= self._max_range
        
        #compute normalization factors
        n = 1/norm.cdf(self._max_range, loc=z_gt[mask], scale=self._sigma_hit)
        
        #compute probabilities
        p1[mask] = n*(1/(np.sqrt(2*np.pi)*self._sigma_hit))*np.exp(-0.5*((z_t[mask]-z_gt[mask])/self._sigma_hit)**2)
                        
        #SHORT PROBABILITY
        #initialize probabilties    
        p2 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t <= z_gt

        #compute normalization factors
        n = 1/(1 - np.exp(-self._lambda_short*z_gt[mask]))
        
        #compute probabilties
        p2[mask] =  n*self._lambda_short*np.exp(-self._lambda_short*z_t[mask])
            
        
        #MAX PROBABILITY
        #initialize probabilties    
        p3 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t >= self._max_range
        
        #compute probabilities
        p3[mask] = 1
        
        #RAND PROBABILITY
        #initialize probabilties    
        p4 = np.zeros(len(z_t))
        
        #mask for non zero probabilities
        mask = z_t < self._max_range
        
        #compute probabilities
        p4[mask] = 1/(self._max_range)
                
        return p1, p2, p3, p4
    
    def get_true_ranges(self, x_t):
        """Find true ranges

        Args:
            x_t (list): state of robot
        """
        return self.ray_casting.get_true_ranges(x_t)

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
                
        #perform ray casting to find GT ranges at various angles
        z_gt = self.ray_casting.get_true_ranges(x_t1)
        
        #sample laser measurements
        z_t1_arr = z_t1_arr[::self.ray_casting._subsampling]
        
        #find probabilities
        p1, p2, p3, p4 = self.sensor_probs(z_t1_arr, z_gt)
        
        #aggregate probabilities
        p = self._z_hit*p1 + \
            self._z_short*p2 + \
            self._z_max*p3 + \
            self._z_rand*p4
        
        #ensure probabilities are non-zero
        p = p + self.eps
        
        #sum of log probabilities
        prob_log = np.sum(np.log(p))
        
        return prob_log


if __name__ == "__main__":
    src_path_map = '/Users/bharath/Documents/acads/spring_2023/16833/hw1/src/data/map/wean.dat'
    map1 = MapReader(src_path_map)

    sm = SensorModel(map1)
    
    t1 = time.time()
    
    xx = 4110
    yy = 5130
    
    for n in range(500):
        z_gt_1 = sm.get_true_ranges([xx, yy, math.pi/2])
    
    t2 = time.time()
        
    print("Vectorized time = ", t2-t1)
    
    print(z_gt_1)
    
    # #sm.rays[yy//10, xx//10] = -5
    # t3 = time.time()
    # for n in range(1):
    #     z_gt_2 = sm.get_true_ranges_vectorized([xx, yy, math.pi/2])
    # t4 = time.time()
    
    #print(z_gt_1[0::10])
    #print(len(z_gt_1))
    # print(z_gt_2[0::10])
    
    # print("Vectorized time = ", t4-t3)
    
    #plt.imshow(sm.map._occupancy_map, 'gray')
    #plt.imshow(sm.rays, 'gray')
    #plt.savefig('./rays.png')
    