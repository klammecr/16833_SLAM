# '''
#     Adapted from course 16831 (Statistical Techniques).
#     Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
#     Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
# '''

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
    def __init__(self, occupancy_map, num_particles):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        # BEST PARAMS SO FAR
        # self._z_hit   = 1.
        # self._z_short = 0.1
        # self._z_max   = 0.5
        # self._z_rand  = 1.5e3

        self._z_hit   = 50.0
        self._z_short = 50.0
        self._z_max   = 50.0
        self._z_rand  = 500

        self._sigma_hit = 75.
        self._lambda_short = 0.1
        
        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000 # cm

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5
        
        #variable to ensure non-zero probabilities
        self.eps = 1e-5
        
        # Store oocupancy map
        self.map = occupancy_map
        
        # Dimensions of occupancy map
        self.h, self.w = self.map._occupancy_map.shape
        
        # Distance that ray can skip during casting iterations
        self.ray_skip_dist = 10
        
        #separation between laser and robot center in cm
        self.laser_loc = 25

        # Store rays
        self.rays = self.map._occupancy_map.copy()
        
        # Small probability value
        self.eps = 1e-5

        self.ray_casting = rc.RayCasting(self.map, num_particles=num_particles)
    
    def p_hit(self, z_t, z_gt):
        #HIT PROBABILITY
        #initialize probabilties
        p1 = np.zeros_like(z_t)
        
        #mask for non zero probabilities
        mask = z_t <= self._max_range
        
        #compute normalization factors
        p1_norm = norm.cdf(self._max_range, loc=z_gt, scale=self._sigma_hit)
        
        # if eta is 0 just mask out because it is pretty much undefined
        mask = np.bitwise_and(mask, p1_norm > 1e-10)
        
        #compute probabilities
        # p1[mask] = n*np.exp(-0.5*((z_t[mask]-z_gt[mask])/self._sigma_hit)**2)
        p1[mask] = norm.pdf(z_t[mask], loc = z_gt[mask], scale = self._sigma_hit) / p1_norm[mask]

        return p1

    def p_short(self, z_t, z_gt):
        #SHORT PROBABILITY
        #initialize probabilties    
        p2 = np.zeros_like(z_t)
        
        #mask for non zero probabilities
        mask = z_t <= z_gt

        #compute normalization factors
        eta = 1/(1 - np.exp(-self._lambda_short*z_gt[mask]))
        
        #compute probabilties
        p2[mask] =  eta * self._lambda_short*np.exp(-self._lambda_short*z_t[mask])

        return p2
    
    def p_max(self, z_t):
        #MAX PROBABILITY
        #initialize probabilties    
        p3 = np.zeros_like(z_t)
        
        #compute probabilities
        p3[z_t == self._max_range] = 1

        return p3

    def p_rand(self, z_t):
        #RAND PROBABILITY
        #initialize probabilties    
        p4 = np.zeros_like(z_t)
        
        #mask for non zero probabilities
        mask = np.bitwise_and(z_t < self._max_range, z_t >= 0)
        
        #compute probabilities
        p4[mask] = 1/(self._max_range)
        
        return p4
    

    def sensor_probs(self, z_t, z_gt):
        """function to compute probabiltiies of measurement
        Args:
            z_t (list): measured data
            z_gt (list): ground truth data
        """
        #convert inputs into numpy arays
        z_t = np.array(z_t)
        z_gt = np.array(z_gt)
        
        # Compute the probabilities for the sensor model
        p1 = self.p_hit(z_t, z_gt)
        p2 = self.p_short(z_t, z_gt)
        p3 = self.p_max(z_t)
        p4 = self.p_rand(z_t)
                     
        return p1, p2, p3, p4
    
    def sensor_probs_vec(self, z_t, z_gt):
        """function to compute probabiltiies of measurement
        Args:
            z_t (list): measured data
            z_gt (list): ground truth data
        """
        # Repeat so we can compare against each particle
        z_t = np.repeat(z_t[np.newaxis,:], z_gt.shape[0], 0)

        # Compute the probabilities for the sensor model
        p1 = self.p_hit(z_t, z_gt)
        p2 = self.p_short(z_t, z_gt)
        p3 = self.p_max(z_t)
        p4 = self.p_rand(z_t)
                     
        return p1, p2, p3, p4
        
    def get_map_with_rays(self):
        return self.ray_casting.rays_map
    
    def get_true_ranges(self, x_t):
        """Find true ranges

        Args:
            x_t (list): state of robot
        """
        return self.ray_casting.get_true_ranges(x_t)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : state belief of all particles [[x_i, y_i, theta_i]] at time t [world_frame]
        param[out] prob_zt1 : probabilities of all particles at time t
        """
        # Clear the rays from last frame
        self.rays = self.map._occupancy_map.copy()
        
        #clip measured ranges
        z_t1_arr = np.clip(z_t1_arr, 0, self._max_range)
        
        #perform ray casting to find GT ranges at various angles
        z_gt = self.ray_casting.get_true_ranges_vec(x_t1)[0]
        
        #sample laser measurements
        z_t1_arr = z_t1_arr[::self.ray_casting._subsampling]

        #find probabilities
        p1, p2, p3, p4 = self.sensor_probs_vec(z_t1_arr, z_gt)
        
        #aggregate probabilities
        p = self._z_hit*p1 + \
            self._z_short*p2 + \
            self._z_max*p3 + \
            self._z_rand*p4
        
        #add small probability
        #p += self.eps #removing this since clipping range will ensure this naturally
        
        #sum of log probabilities
        prob_log = np.sum(np.log(p), axis = 1)

        # apply softmax to normalize the probabilities
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/      
        a = prob_log.max()
        logsumexp = a + np.log(np.sum(np.exp(prob_log - a)))
        prob = np.exp(prob_log - logsumexp)
    
        # #my implementation
        # max_val = prob_log.max()
        # prob_log = prob_log - max_val
        # prob = np.exp(prob_log)
        # prob = prob/np.sum(prob)
        
        
        return prob


if __name__ == "__main__":
    pass
    # src_path_map = '/Users/bharath/Documents/acads/spring_2023/16833/hw1/src/data/map/wean.dat'
    # map1 = MapReader(src_path_map)

    # sm = SensorModel(map1)
    
    # t1 = time.time()
    
    # xx = 4110
    # yy = 5130
    
    # for n in range(500):
    #     z_gt_1 = sm.get_true_ranges([xx, yy, math.pi/2])
    
    # t2 = time.time()
        
    # print("Vectorized time = ", t2-t1)
    
    # print(z_gt_1)
    
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

    # Chris Klamemr Debug