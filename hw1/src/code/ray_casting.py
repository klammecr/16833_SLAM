
from multiprocessing import Pool, Process, Queue
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader

class RayCasting():
    def __init__(self, occupancy_map):
        """constructor for RayCating class

        Args:
            occupancy_map (_type_): occupancy map object
        """
        
        # Maximum range for ray casting
        self._max_range = 750.

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5
                
        # Store oocupancy map
        self.map = occupancy_map._occupancy_map
        
        # Store resolution of occupancy map
        self.resolution = occupancy_map._resolution
        
        # Dimensions of occupancy map
        self.h, self.w = self.map.shape
        
        # Distance that ray can skip during casting iterations
        self.ray_skip_dist = 1
        
        #separation between laser and robot center in cm
        self.laser_loc = 25

        # Store rays
        self.rays_map = self.map.copy()

        #perform ray casting relative to robot
        self.rays = self.relative_ray_casting()

    def relative_ray_casting(self):
        """Perform ray casting relative to robot's position

        Returns:
            np.ndarray: array containing x,y values of each point along each ray
        """
        #angles at which rays are cast
        angles = np.arange(1, 181, self._subsampling)
        
        #distances at which z is computed along each ray
        diag_length = math.ceil(math.sqrt(self.h**2 + self.w**2))
        dists = np.arange(0, diag_length, 1)
        
        #number of angles and points along each ray
        num_angles = angles.size
        num_points = dists.size
                
        #create array to store x and y for each point in ray
        rays = np.zeros((num_angles, num_points, 2))
        
        #perform ray casting relative to robot's location
        for i, a in enumerate(angles):
            for j, d in enumerate(dists):
                x = d*np.cos((np.pi/180)*a)
                y = d*np.sin((np.pi/180)*a)
                rays[i, j] = [x, y]
        
        #scale distances
        rays = self.resolution*rays
        
        return rays
        
    
    def transform_rays(self, x_t):
        """Transform pre-computed ray coordinates according to robot's state

        Args:
            x_t (list): list containing position and orientation of robot
        """
        
        #unpack state
        x, y, angle = x_t
        
        #adjust angle
        angle = angle - np.pi/2
        
        #make a copy of rays
        rays = self.rays.copy()
        
        #rotate the rays
        rot_mat = [[np.cos(angle), -np.sin(angle)],\
                    [np.sin(angle), np.cos(angle)]]
        
        rays = np.matmul(rays, rot_mat)
        
        #translate the rays 
        rays[:,:,0] = rays[:,:,0] + x
        rays[:,:,1] = rays[:,:,1] + y
        
        return rays
    
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
    
    def get_true_ranges(self, x_t):
        """Find true depths at all angles for a given robot state

        Args:
            x_t (list): state of robot
        """
        
        #adjust robot's state into laser's state
        x_t = self.sensor_location(x_t)
        
        #adjust rays based on robot's state
        rays = self.transform_rays(x_t)
                
        #find obstacle along each ray
        num_angles, num_dists = rays.shape[:2]
        
        #array to store ranges
        z_true = np.zeros((num_angles))
        
        #convert cm to px
        x_int = np.round(rays[:,:,0]/self.resolution).astype(np.int32)
        y_int = np.round(rays[:,:,1]/self.resolution).astype(np.int32)
        
        #filter coordinates outside map
        m_lx = x_int >= 0
        m_ly = y_int >= 0
        m_hx = x_int < self.w
        m_hy = y_int < self.h
        
        m_filter = m_lx*m_ly*m_hx*m_hy
        m_filter = np.logical_not(m_filter)
                
        #set invalid coordinates to 0 to avoid errors
        x_int[m_filter] = 0
        y_int[m_filter] = 0
        
        #find coordinates that hit the obstacle
        m_uk = self.map[y_int, x_int] != -1
        m_free = self.map[y_int, x_int] <= self._min_probability
        
        #take intersection of filter and obstacle masks
        m_overall = np.logical_not(np.logical_not(m_filter)*m_uk*m_free)
        
        #convert mask into int
        m_overall = m_overall.astype('int')
        
        #take cumulative sum along each row
        m_overall_cumsum = np.cumsum(m_overall, axis=1)
        
        #find pixels where cumsum = 1
        angs, dists = np.where(m_overall_cumsum == 1)
        
        #populate z_true based on x and y
        for (a, d) in zip(angs, dists):
            z_true[a] = d
                
        # #convert cm to px
        # x_int_arr = np.round(rays[:,:,0]/self.resolution).astype(np.int32)
        # y_int_arr = np.round(rays[:,:,1]/self.resolution).astype(np.int32)
        
        # for a in range(num_angles):
        #     for d in range(num_dists):
        #         #unpack rays
        #         x, y = rays[a, d]
                
        #         #convert cm into pixels
        #         x_int = x_int_arr[a, d]
        #         y_int = y_int_arr[a, d]
                
        #         #check for obstacles
        #         #if ray goes beyond edge or hits a wall
        #         if x_int < 0 or x_int > self.w - 1 or \
        #             y_int < 0 or y_int > self.h - 1 or \
        #             self.map[y_int][x_int] == -1 or \
        #             self.map[y_int][x_int] > self._min_probability:
        #                 #set depth
        #                 z_true[a] = d
        #                 break
                
        #         #mark coordinate on map
        #         self.rays_map[y_int, x_int] = -2
        
        #scale range
        z_true = z_true*self.resolution
        
        return z_true


if __name__ == "__main__":
    src_path_map = './../data/map/wean.dat'
    map1 = MapReader(src_path_map)

    t1 = time.time()
    rc = RayCasting(map1)
    t2 = time.time()
    print("Time taken to initialize = {} seconds".format(t2-t1))
    
    xx = 4110
    yy = 5130
    t3 = time.time()
    for n in range(1):
        z_true = rc.get_true_ranges([xx, yy, math.pi/2])
    
    t4 = time.time()
    print("Time taken to compute depth = {} seconds".format(t4-t3))
    
    print(z_true)
    
    #plt.imshow(sm.map._occupancy_map, 'gray')
    plt.imshow(rc.rays_map, 'gray')
    plt.savefig('./rays_new.png')