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
from datetime import datetime
from adaptive_particles import AdaptiveParticleCalculator

def get_time_string():
    now = datetime.now()

    # Get the year, month, day, hour, minute, and second as integers
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)

    # Combine the integers into a string in the desired format
    formatted_date = year + month + day + hour + minute + second
    
    return formatted_date

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
    angles = np.random.uniform(-np.pi, np.pi, (num_particles, 1))
    
    #initialize weights
    weights = np.ones((num_particles, 1), dtype=np.float64)
    weights = weights / num_particles
    
    #stack all particle attributes
    X_bar_init = np.hstack((x_scaled.reshape(-1, 1), y_scaled.reshape(-1, 1), \
                            angles.reshape(-1, 1), weights.reshape(-1, 1)))
    
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
    import cProfile
    import sys
    pr = cProfile.Profile()
    pr.enable()

    np.random.seed(5875)
    parser = argparse.ArgumentParser()
    working_dir = os.getcwd()
    parser.add_argument('--path_to_map', default=f'../data/map/wean.dat')
    parser.add_argument('--path_to_log', default=f'../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--z_hit',   default=100, type=float)
    parser.add_argument('--z_short', default=50, type=float)
    parser.add_argument('--z_max',   default=50, type=float)
    parser.add_argument('--z_rand',  default=100000, type=float)
    parser.add_argument('--alpha1',  default=1e-4, type=float)
    parser.add_argument('--alpha2',  default=1e-4, type=float)
    parser.add_argument('--alpha3',  default=7.5e-4, type=float)
    parser.add_argument('--alpha4',  default=7.5e-4, type=float)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
        
    if args.video:
        #add file path to output
        tag = args.path_to_log.split('/')[-1].split('.')[0] + \
                '_' + get_time_string() + f"ZH={args.z_hit}_ZS={args.z_short}_ZM={args.z_max}_ZR={args.z_rand}" + \
                f"a1={args.alpha1}_a2={args.alpha2}_a3={args.alpha3}_a4={args.alpha4}" + '.mp4'
        args.output = os.path.join(args.output, tag)

        print('Output video will be saved to  {}'.format(args.output))

    # Load in occupancy map
    src_path_map  = args.path_to_map
    src_path_log  = args.path_to_log
    map_obj       = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile       = open(src_path_log, 'r')

    # Setup the visualizer
    if args.visualize:
        vis = Visualizer(occupancy_map, args.output, video = args.video)

    # Create init particles
    num_particles = args.num_particles
    X_bar = init_particles_freespace(num_particles, map_obj)

    # Create necessary objects for motion, sensor, and resampling
    motion_model = MotionModel(args)
    sensor_model = SensorModel(args, map_obj, num_particles)
    resampler    = Resampling()
    adaptive_particles = AdaptiveParticleCalculator(num_particles)
    sort_idxs = None
    
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    first_time_idx = True
    
    #keep track of time
    t1 = time.time()
    
    for time_idx, line in enumerate(logfile):
        ## READ DATA
        # read the type of log file
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # read odometry and timestamp data
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        print(f"Time {time.time()-t1} sec :Processing time step {time_idx} at time {time_stamp}")
        # read sensor data
        if (meas_type == "L"):
            odometry_laser = meas_vals[3:6]
            ranges = meas_vals[6:-1]

        # for first time step, initialize odometry readings
        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue
        
        ##PERFORM PARTICLE FILTERING
        # array to store new particles
        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        
        #set input variables to particle filter
        # motion parameters
        x_t0 = X_bar[:, :3]
        u_t1 = odometry_robot

        # sensor parameters
        z_t  = ranges
        
        # step 1: apply motion model
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)

        # step 2: apply sensor model
        if (meas_type == "L"):
            w_t  = sensor_model.beam_range_finder_model(z_t, x_t1, sort_idxs)
        else:
            if sort_idxs is None:
                w_t = X_bar[:, 3]
            else:
                w_t = X_bar[sort_idxs, 3]

        # aggregate data for new particles
        X_bar_new = np.hstack((x_t1, w_t.reshape(-1, 1)))

        # step 3: resample particles 
        # Only resample when there is motion
        if np.sum(np.abs(x_t1 - x_t0)) > 1e-10:
            # X_bar = resampler.multinomial_sampler(X_bar)
            X_bar = resampler.low_variance_sampler(X_bar_new)
            #X_bar, sort_idxs = adaptive_particles.calculate_naive(X_bar)
        else:
            X_bar = X_bar_new.copy()
            sort_idxs = None
            
        # visualize map
        vis.visualize_timestep(X_bar, time_stamp, sort_idxs)

        # reset parameters for next iteration
        u_t0 = u_t1
    
    # Explicitly delete the visualizer, just in case
    del vis
    
    pr.disable()
    pr.print_stats()