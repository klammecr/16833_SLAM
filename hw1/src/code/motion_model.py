'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
from math import atan2, cos, sin, sqrt, abs


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        # Output vector for the pose at time t
        x_t1 = np.zeros_like(x_t0)

        # Find the rotation from time (t-1) to the centroid at time t from odometry measurements
        delta_rot_1 = atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2] # Angle between the translation vector and the first angle measurement
        delta_trans = sqrt((u_t1[0] - u_t0[0]) ** 2 + (u_t1[1] - u_t0[1]) ** 2) 
        delta_rot_2 = u_t1[2] - u_t0[2] - delta_rot_1 # This is just the angle between the translation vector and the second angle measurement

        # Sample the motion noise and remove it from the delta values
        delta_rot_1 -= np.random.normal(loc = 0.0, scale = self._alpha1 * abs(delta_rot_1) + self._alpha2 * abs(delta_trans))
        delta_trans -= np.random.normal(loc = 0.0, scale = self._alpha3 * abs(delta_trans) + self._alpha4 * abs(delta_rot_1 + delta_rot_2))
        delta_rot_2 -= np.random.normal(loc = 0.0, scale = self._alpha1 * abs(delta_rot_2) + self._alpha2 * abs(delta_trans))

        # Estimate the pose of the robot at time t
        x_t1[0] = x_t0[0] + delta_trans * cos(x_t0[2] + delta_rot_1)
        x_t1[1] = x_t0[1] + delta_trans * sin(x_t0[2] + delta_rot_1)
        x_t1[2] = x_t0[2] + delta_rot_1 + delta_rot_2
        
        return x_t1
