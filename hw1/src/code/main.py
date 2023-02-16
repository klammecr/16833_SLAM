'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os
import cv2 as cv2

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


# def visualize_timestep(X_bar, tstep, output_path):
#     x_locs = X_bar[:, 0] / 10.0
#     y_locs = X_bar[:, 1] / 10.0
#     scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
#     plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
#     plt.pause(0.00001)
#     scat.remove()


def visualize_timestep(X_bar, tstep, occupancy_map, resolution=10.0):
    #compute coordiantes
    x_locs = X_bar[:, 0] // resolution
    y_locs = X_bar[:, 1] // resolution

    x_locs = x_locs.astype('int')
    y_locs = y_locs.astype('int')
    
    #deep copy of map
    occ_map = occupancy_map.copy()
    
    #scale and convert map to 3 channels
    #scale
    occ_map[occ_map < 0] = 0
    occ_map = (occ_map - occ_map.min())/(occ_map.max() - occ_map.min())
    occ_map = (occ_map*255).astype('uint8')
    
    #stack
    occ_map = occ_map[:,:,None]
    occ_map = np.stack((occ_map, occ_map, occ_map), axis=-1)
    
    #add particles to map
    occ_map[y_locs, x_locs, -1] = 255
    
    return occ_map

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
    TODO : Add your code here
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
    X_bar_init = np.hstack((x_scaled, y_scaled, angles, weights))
    
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
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    X_bar = init_particles_random(num_particles, occupancy_map)
    # X_bar = init_particles_freespace(num_particles, occupancy_map)
    
    #initialize video writer
    video_writer = None
    if args.video:
        #format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        #resolution of frames
        dim = occupancy_map.shape[0]
        
        #video writer instance
        video_writer = cv2.VideoWriter(os.path.join(args.output, 'output.mp4'), fourcc, (dim, dim))        
    
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

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

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
                
            # else:
            #     X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        #convert log probabilities into probabilities
        prob_log = X_bar_new[:,-1]
        
        #apply softmax
        prob_log = prob_log -  prob_log.max()
        prob = (np.exp(prob_log))/(np.sum(np.exp(prob_log)))
        
        #add probabilities to particle parameters
        X_bar_new[:,-1] = prob
        
        
        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            map_vis = visualize_timestep(X_bar, time_idx, occupancy_map, map_obj._resolution)
            cv2.imshow('Output', map_vis)
            if args.video:
                video_writer.write(map_vis)
    
    if args.visualize:
        cv2.destroyAllWindows()
        if args.video:
            video_writer.release()