'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
from math import atan2, sqrt


def wrap_angle(angle):
    """All of our angles have to be between -pi and pi

    Args:
        angle (float): The angle to warp

    Returns:
            float: The wrapped angle of range [-pi, pi)
    """
    # Put the range to [0, 360) then take the modulo in case we go over or under
    angle_wrap = (angle + np.pi) % (2*np.pi) - np.pi    
    return angle_wrap


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 1e-4
        self._alpha2 = 1e-4
        self._alpha3 = 7.5e-4
        self._alpha4 = 7.5e-4

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        # See if there is consequential motion
        # if np.sum(np.abs(u_t1 - u_t0)) <= 1e-10:
        #     return x_t0

        # Output vector for the pose at time t
        x_t1 = np.zeros_like(x_t0)

        # Find the rotation from time (t-1) to the centroid at time t from odometry measurements
        delta_rot_1 = wrap_angle(atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]) # Angle between the translation vector and the first angle measurement
        delta_trans = sqrt((u_t1[0] - u_t0[0]) ** 2 + (u_t1[1] - u_t0[1]) ** 2) 
        delta_rot_2 = wrap_angle(u_t1[2] - u_t0[2] - delta_rot_1) # This is just the angle between the translation vector and the second angle measurement

        # Remove the independent noise
        delta_rot_1_var = self._alpha1 * delta_rot_1**2 + self._alpha2 * delta_trans**2
        delta_trans_var = self._alpha3 * delta_trans**2 + self._alpha4 * (delta_rot_1**2 + delta_rot_2**2)
        delta_rot_2_var = self._alpha1 * delta_rot_2**2 + self._alpha2 * delta_trans ** 2
        delta_rot_1 -= wrap_angle(np.random.normal(0.0, delta_rot_1_var**0.5, size = x_t0.shape[0]))
        delta_trans -= np.random.normal(0.0, delta_trans_var**0.5, size = x_t0.shape[0])
        delta_rot_2 -= wrap_angle( np.random.normal(0.0, delta_rot_2_var**0.5, size = x_t0.shape[0]))

        # Estimate the pose of the robot at time t
        x_t1[:, 0] = x_t0[:, 0] + delta_trans * np.cos(x_t0[:, 2] + delta_rot_1)
        x_t1[:, 1] = x_t0[:, 1] + delta_trans * np.sin(x_t0[:, 2] + delta_rot_1)
        x_t1[:, 2] = wrap_angle(x_t0[:, 2] + delta_rot_1 + delta_rot_2)
        
        return x_t1