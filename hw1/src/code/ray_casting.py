
from multiprocessing import Pool, Process, Queue
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader

class RayCasting():
    def __init__(self, occupancy_map, num_particles):
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
        self.num_particles = num_particles
        self.rays = self.relative_ray_casting()


    def relative_ray_casting(self):
        """Perform ray casting relative to robot's position

        Returns:
            np.ndarray: array containing x,y values of each point along each ray
        """
        #angles at which rays are cast
        # Take the angles going counter-clockwise
        angles = np.arange(-90, 90, self._subsampling)
        
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
        
        # Allocate rays for every particle
        rays = np.tile(rays, (self.num_particles, 1, 1, 1))

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
        
        #make a copy of rays for all particles
        rays     = self.rays.copy()
        out_rays = np.zeros_like(rays)
        
        #rotate the rays
        # rot_mat = [[np.cos(angle), -np.sin(angle)],\
        #             [np.sin(angle), np.cos(angle)]]
        # rays = np.matmul(rays, rot_mat)
        #translate the rays 
        #rays[:, :, :, 0] += x
        #rays[:, :, :, 1] += y
        
        # Vectorized implementation for rotation from the world frame to the robot frame
        #rotate the rays
        # transform = np.tile(np.eye(3), (1, self.num_particles)).reshape(3, 3, self.num_particles)
        # transform[0, 0, :] = np.cos(angle)
        # transform[1, 1, :] = np.cos(angle)
        # transform[1, 0, :] = np.sin(angle)
        # transform[0, 1, :] = -np.sin(angle)
        # transform[2, 0, :] = x # translation by x
        # transform[2, 1, :] = y # translation by y
        # Vectorized matrix multiplication
        # rays = np.dot(rays, transform)
        # rays = np.moveaxis(rays, -1, 0)

        # Sizes for everything
        particles, angles, points, _ = rays.shape
        angle = angle.reshape(particles, 1)
        x     = x.reshape(particles, 1)
        y     = y.reshape(particles, 1)

        # Create an angle array to transform the correct points by the correct angles
        angle_arr = np.repeat(angle, angles*points, axis = 1).reshape(particles, angles, points)
        x_trans_arr = np.repeat(x, angles*points, axis = 1).reshape(particles, angles, points)
        y_trans_arr = np.repeat(y, angles*points, axis = 1).reshape(particles, angles, points)

        # Rotate and translate the ray relative to the robot's body frame
        out_rays[:, :, :, 0] = (rays[:, :, :, 0] *  np.cos(angle_arr) + rays[:, :, :, 1] * -np.sin(angle_arr)) + x_trans_arr
        out_rays[:, :, :, 1] = (rays[:, :, :, 0] *  np.sin(angle_arr) + rays[:, :, :, 1] * np.cos(angle_arr))  + y_trans_arr

        return out_rays
    
    def sensor_location(self, x):
        """Find sensor's position based on state of robot
        Args:
            x (list): state of robot represented by a particle
        """
        #get x, y, theta of robot from state
        rx    = x[:, 0]
        ry    = x[:, 1]
        theta = x[:, 2]
        
        #compute laser's location
        lx = rx + self.laser_loc*np.cos(theta) 
        ly = ry + self.laser_loc*np.sin(theta)
        
        return [lx, ly, theta]
    
    def get_true_ranges_vec(self, x_t):
        """Find true depths at all angles for a given robot state

        Args:
            x_t (list): state of robot
        """
        
        #adjust robot's state into laser's state
        x_t = self.sensor_location(x_t)
        
        #adjust rays based on robot's state
        rays = self.transform_rays(x_t)
                
        #find obstacle along each ray
        num_angles = rays.shape[1]
        
        #array to store ranges
        z_true = np.zeros((self.num_particles, num_angles))
        
        #convert cm to px
        x_int = np.round(rays[:,:,:,0]/self.resolution).astype(np.int32)
        y_int = np.round(rays[:,:,:,1]/self.resolution).astype(np.int32)
        
        #filter coordinates outside map
        m_lx = x_int < 0
        m_ly = y_int < 0
        m_hx = x_int >= self.w
        m_hy = y_int >= self.h
        
        # Filter out pixels on or beyond the boundary
        m_filter = np.logical_or.reduce((m_lx, m_ly, m_hx, m_hy))
                
        # Clip x and y coordinates to their max and min values
        x_int[m_filter] = 0
        y_int[m_filter] = 0
        
        #find coordinates that hit the obstacle
        obstacles = np.bitwise_or(self.map[y_int, x_int] == -1, self.map[y_int, x_int] >= self._min_probability)
        
        #take intersection of filter and obstacle masks
        # This is all the occupied areas of the map
        occupied = np.bitwise_or(m_filter, obstacles).astype("int")
        
        # Travel along each ray for a given particle and angle once we hit a point that is intraversible, our cumsum will be 1
        m_overall_cumsum = np.cumsum(occupied, axis=2)
        
        # Look at the particle, angle, and distance indexes to look for the boundaries
        particles, angs, dists = np.where(m_overall_cumsum == 1)
        
        #populate z_true based on x and y
        z_true[particles, angs] = dists
        # for (p, a, d) in zip(particles, angs, dists):
        #     z_true[p, a] = d
                
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
        #         if x_int < 0 or x_int > self.w - 1 or /
        #             y_int < 0 or y_int > self.h - 1 or /
        #             self.map[y_int][x_int] == -1 or /
        #             self.map[y_int][x_int] > self._min_probability:
        #                 #set depth
        #                 z_true[a] = d
        #                 break
                
        #         #mark coordinate on map
        #         self.rays_map[y_int, x_int] = -2
        
        #scale range
        z_true *= self.resolution
        
        return z_true, x_int, y_int


if __name__ == "__main__":
    # src_path_map = './../data/map/wean.dat'
    # map1 = MapReader(src_path_map)

    # t1 = time.time()
    # rc = RayCasting(map1)
    # t2 = time.time()
    # print("Time taken to initialize = {} seconds".format(t2-t1))
    
    # xx = 4110
    # yy = 5130
    # t3 = time.time()
    # for n in range(1):
    #     z_true = rc.get_true_ranges([xx, yy, math.pi/2])
    
    # t4 = time.time()
    # print("Time taken to compute depth = {} seconds".format(t4-t3))
    
    # print(z_true)
    
    # #plt.imshow(sm.map._occupancy_map, 'gray')
    # plt.imshow(rc.rays_map, 'gray')
    # plt.savefig('./rays_new.png')

    # Visualization of the map for Chris Klammer
    src_path_map = 'C:/Users/chris/dev/16833/hw1/src/data/map/wean.dat'
    map1 = MapReader(src_path_map)
    occ_map = map1.get_map().copy()
    occ_map[occ_map < 0] = 0
    occ_map = (occ_map - occ_map.min())/(occ_map.max() - occ_map.min())
    occ_map = (occ_map*255).astype('uint8')
    # Stack the BGR channels
    occ_map = np.stack((occ_map, occ_map, occ_map), axis=-1)

    # Location of the robot
    xx = 4110
    yy = 5130

    # Create a circle for the robot position
    import cv2

    # Create a fixture robot position
    num_particles = 3
    robot_pos = np.zeros((num_particles, 3))

    # CUSTOM PARTICLES FOR DEBUGGING
    robot_pos[0, 0] = xx # x pos
    robot_pos[0, 1] = yy # y pos
    robot_pos[0, 2] = np.pi/2  # angle of robot
    # Optional Particle 2
    robot_pos[1, 0] = 6200 # x pos
    robot_pos[1, 1] = 1500 # y pos
    robot_pos[1, 2] = -np.pi/4  # angle of robot
    # Optional particle 3 out of the map
    robot_pos[2, 0] = 1000 # x pos
    robot_pos[2, 1] = 1000 # y pos
    robot_pos[2, 2] = 0  # angle of robot

    # Create circles for each particle
    for particle in range(robot_pos.shape[0]):
        x = robot_pos[particle, 0]
        y = robot_pos[particle, 1]
        cv2.circle(occ_map, (int(x//10), int(y//10)), radius = 3, color = (0,0,255), thickness = 5)

    # Perform ray casting for the robot at the current location
    t1 = time.time()
    rc = RayCasting(map1, num_particles)
    t2 = time.time()
    print(f"Time taken to initialize = {t2-t1} seconds")

    # VALIDATED TEST TRANSFORM RAYS
    # x_t = rc.sensor_location(robot_pos)
    # rc.transform_rays(x_t)

    # TEST GET TRUE RANGES
    t3 = time.time()
    z_true, x_int, y_int = rc.get_true_ranges_vec(robot_pos)
    t4 = time.time()
    print(f"Time to run vectorized get true ranges for {num_particles} particles: {t4-t3} seconds")

    # DEBUG THE LASERS
    occ_map[y_int, x_int, 1] = 255

    # DEBUG THE RANGES
    for i in range(z_true.shape[0]):
        for ang_idx, ang in enumerate(range(-90, 90, rc._subsampling)):
            p2_x = robot_pos[i, 0] + z_true[i, ang_idx] * np.cos(robot_pos[i, 2] + ang * np.pi/180)
            p2_y = robot_pos[i, 1] + z_true[i, ang_idx] * np.sin(robot_pos[i, 2] + ang * np.pi/180)
            # Convert from cm to pix
            p1_x = robot_pos[i, 0] // 10
            p1_y = robot_pos[i, 1] // 10
            p2_x /= 10
            p2_y /= 10

            p1 = (int(p1_x), int(p1_y))
            p2 = (int(p2_x), int(p2_y))
            cv2.line(occ_map, p1, p2, color = (0,0,255))

    # Display the map with the robot and the range
    cv2.imshow("DEBUG OCCUPANCY MAP", occ_map)
    cv2.waitKey()
    pass