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

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 1
        
        # Number of processes to be used for subsampling
        self.num_processes = 8
        
        # Store oocupancy map
        self.map = occupancy_map
        
        # Dimensions of occupancy map
        self.h, self.w = self.map._occupancy_map.shape
        
        # Distance that ray can skip during casting iterations
        self.ray_skip_dist = 10
        
        #separation between laser and robot center in cm
        self.laser_loc = 25

        #variable to ensure non-zero probabilities
        self.eps = 1e-5
        
        # Store rays
        self.rays = self.map._occupancy_map.copy()
        
    def sensor_location(self, x):
        """Find sensor's position based on state of robot

        Args:
            x (list): state of robot represented by a particle
        """
        #get x, y, theta of robot from state
        rx, ry, theta = x
        
        #compute laser's location
        lx = rx + self.laser_loc*np.cos(theta) 
        ly = ry + self.laser_loc*np.sin(theta)
        
        return [lx, ly, theta]
    
    def ray_casting_vectorized(self, arg_list, results):
        """vectorized implementation of ray casting

        Args:
            args (Queue): Queue with input arguments
            results (Queue): Queue that stores results
        """
        # while True:
        #     #sample and input
        #     try:
        #         x, angle = args.get_nowait()
        #     except queue.Empty:
        #         break
        #     #if sampling is successful, perform ray casting
        #     else:
        #         #cast a ray
        #         z_gt_a = self.ray_casting(x, angle)
        #         results.put([angle, z_gt_a])

        #iteratively process inputs and cast rays
        for a in arg_list:
            x, angle = a
            z_gt_a = self.ray_casting(x, angle)
            results.put(z_gt_a)
         
    def ray_casting(self, x, angle):
        """ray casting algorithm to find true range 

        Args:
            x (list): state of the range sensor represented by a particle
            angle (float): angle of laser beam
        """
        
        #unpack particle
        lx, ly, ltheta = x
        
        #measure angle of laser beam in global reference frame
        theta = (ltheta + angle) - math.pi/2
        
        #perform ray casting
        #initialize positions to positions of range sensor
        cx, cy = lx, ly
        
        #iterate till wall is hit or the ray reaches edge of map
        while True:
            #convert cm into px
            cx_p = math.floor(cx/self.map._resolution)
            cy_p = math.floor(cy/self.map._resolution)
            
            #if ray goes beyond edge
            if cx_p < 0 or cx_p > self.w - 1 or \
                cy_p < 0 or cy_p > self.h - 1:
                    break
            
            #if ray hits the wall
            if self.map._occupancy_map[cy_p][cx_p] != 0:
                break
            
            #update x and y coordinates
            cx = cx + self.ray_skip_dist*np.cos(theta)
            cy = cy + self.ray_skip_dist*np.sin(theta)
            
            self.rays[cy_p][cx_p] = -2
            #print("ray = ", cy, cx, self.rays[cy_p][cx_p])
            
        z_gt = math.sqrt((cx-lx)**2 + (cy-ly)**2)

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
            z_gt_angle = self.ray_casting(x, i*(math.pi/180))
                        
            #add range to list
            z_gt.append(z_gt_angle)
        
        return z_gt
    
    def get_true_ranges_vectorized(self, x):
        """function to find true range for a given state(x,y,theta) of robot using vectorized implementation

        Args:
            x (list): state of range sensor
        """
        #array to store true ranges
        z_gt = []
        
        #store angles at which rays are cast
        angles = np.arange(1, 181, self._subsampling)
        
        #create list to store input arguments
        args = []
        
        #create Queue to store results
        results = Queue()
        for a in angles:
            args.append([x, a*(math.pi/180)])
        
        #split inputs into chunks
        chunk_size = len(args)//self.num_processes
        args_list = [args[i:i+chunk_size] for i in range(0, len(args), chunk_size)]
        
        #start processes
        processes = []
        for a in args_list:
            p = Process(target = self.ray_casting_vectorized, args=(a, results,))
            processes.append(p)
            p.Daemon = True
            p.start()
            print('Started process', len(a))

        #end processes
        for a in args_list:
            p.join()
        
        #unpack results
        while not results.empty():
            print(results.get())
        #print(results)
        # z_gt = [0 for i in range(angles.size)]
        
        # while not results.empty():
        #     a, z = results.get()
            
        # return z_gt
        return [0]
    
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
        mask = z_t > self._max_range
        
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
        
        #sample laser measurements
        z_t1_arr = z_t1_arr[::self._subsampling]
        
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
    src_path_map = './../data/map/wean.dat'
    map1 = MapReader(src_path_map)

    sm = SensorModel(map1)
    
    t1 = time.time()
    xx = 4110
    yy = 5130
    for n in range(1):
        z_gt_1 = sm.get_true_ranges([xx, yy, math.pi/2])
        t2 = time.time()
        
    print(t2-t1)
    
    #sm.rays[yy//10, xx//10] = -5
    t3 = time.time()
    for n in range(1):
        z_gt_2 = sm.get_true_ranges_vectorized([4110, 5130, math.pi/2])
    t4 = time.time()
    
    #print(z_gt_1[0::10])
    #print(len(z_gt_1))
    # print(z_gt_2[0::10])
    
    print(t4-t3)
    
    #plt.imshow(sm.map._occupancy_map, 'gray')
    plt.imshow(sm.rays, 'gray')
    plt.savefig('./rays.png')
    