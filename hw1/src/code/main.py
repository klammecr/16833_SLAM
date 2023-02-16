'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
# Third Party
import argparse
import numpy as np
import sys, os
import cv2 as cv2
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

# In House
from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from visualizer import Visualizer

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))
    
    #find empty space
    y_free , x_free = np.where(occupancy_map._occupancy_map == 0)
    
    #shuffle indices
    idx = np.random.permutation(y_free.size)
    
    #sample N indices
    idx_sampled = idx[0:num_particles]
    
    #sample unscaled coordinates
    y_unscaled = y_free[idx_sampled]
    x_unscaled = x_free[idx_sampled]
    
    #randomly generate values between 0 and 10 to account for map scale
    y_perturb = np.random.uniform(0, occupancy_map._resolution, y_unscaled.shape)
    x_perturb = np.random.uniform(0, occupancy_map._resolution, x_unscaled.shape)
    
    #generate scaled coordinates
    y_scaled = y_unscaled*occupancy_map._resolution + y_perturb
    x_scaled = x_unscaled*occupancy_map._resolution + x_perturb

    #generate random orientations
    angles = np.random.uniform(-3.14, 3.14, (num_particles, 1))
    
    #initialize weights
    weights = np.ones((num_particles, 1), dtype=np.float64)
    weights = weights / num_particles
    
    #stack all particle attributes
    X_bar_init = np.hstack((x_scaled.reshape(-1, 1), y_scaled.reshape(-1, 1), angles.reshape(-1, 1), weights.reshape(-1, 1)))
    
    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='C:/Users/chris/dev/16833/hw1/src/data/map/wean.dat')
    parser.add_argument('--path_to_log', default='C:/Users/chris/dev/16833/hw1/src/data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load in occupancy map
    src_path_map  = args.path_to_map
    src_path_log  = args.path_to_log
    map_obj       = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile       = open(src_path_log, 'r')

    # Create necessary objects for motion, sensor, and resampling
    motion_model = MotionModel()
    sensor_model = SensorModel(map_obj)
    resampler    = Resampling()

    # Setup the visualizer
    if args.visualize:
        vis = Visualizer(occupancy_map, args.output, video = args.video)

    # Create init particles
    num_particles = args.num_particles
    X_bar = init_particles_freespace(num_particles, map_obj)     
    
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        print(f"Processing time step {time_idx} at time {time_stamp}")

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Vectorize it
        z_t  = ranges
        
        """ MOTION MODEL """
        x_t0 = X_bar[:, :3]
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        # w_t  = sensor_model.beam_range_finder_model_vec(z_t, x_t1)

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                # Find the log probabilities
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1[m])
                
                X_bar_new[m, :] = np.hstack((x_t1[m], w_t))
            else:
                X_bar_new[m, :] = np.hstack((x_t1[m], X_bar[m, 3]))

        #convert log probabilities into probabilities
        prob_log = X_bar_new[:,-1]

        # TODO: TEMPORARY HACK, WE NEED TO VECTORIZE TO MAKE THIS RUN SMOOTH
        # THIS IS HERE FOR THE ELSE CASE WHEN WE DONT GET LOG PROBS
        if np.sum(prob_log) != 1.0:
            #apply softmax
            prob_log = prob_log -  prob_log.max()
            prob = (np.exp(prob_log))/(np.sum(np.exp(prob_log)))
            
            #add probabilities to particle parameters
            X_bar_new[:,-1] = prob

        # Add probabilities to particle parameters
        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        # Only resample when there is motion
        if np.sum(np.abs(x_t1 - x_t0)) > 1e-10:
            # X_bar = resampler.multinomial_sampler(X_bar)
            X_bar = resampler.low_variance_sampler(X_bar)

        # Visualize
        vis.visualize_timestep(X_bar, time_stamp)

    # Explicitly delete the visualizer, just in case
    del vis