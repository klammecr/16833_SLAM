'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def wrap2pi(angle_rad):
    """
    Wrap an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    # Phase shift to bring to 2pi, deal with wrap around then undo phase shift
    angle_wrap = ((angle_rad + np.pi) % (2*np.pi)) - np.pi   
    return angle_wrap

def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    Initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''
    # Init/helper variables
    k            = init_measure.shape[0] // 2
    landmark     = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))

    # Extract out the initial bearing
    init_betas = init_measure[0::2].squeeze()

    # Extract out the initial ranges
    init_ranges = init_measure[1::2].squeeze()

    # Impart noise on the pose
    noisy_px  = np.random.normal(init_pose[0], init_pose_cov[0,0])
    noisy_py  = np.random.normal(init_pose[1], init_pose_cov[1,1])
    noisy_pth = np.random.normal(init_pose[2], init_pose_cov[2,2])

    # Calculate landmarks x & y component
    l_x = noisy_px + init_ranges * np.cos(noisy_pth + init_betas)
    l_y = noisy_py + init_ranges * np.sin(noisy_pth + init_betas)

    # Put the landmrks in the np array
    landmark[0::2] = np.reshape(l_x, (-1, 1))
    landmark[1::2] = np.reshape(l_y, (-1, 1))

    # Calculate landmarks covariance matrix (deviation of the landmarks from the mean)
    for i in range(k):
        # We take the function (say h) that transforms measurements to locations in the world
        # We will calculate the following jacobian:
        # [dh_x/d_beta, dh_x/d_r]
        # [dh_y/d_beta, dh_y/d_r]
        d_beta_x = -init_ranges[i] * np.sin(init_pose[2] + init_betas[i])
        d_beta_y =  init_ranges[i] * np.cos(init_pose[2] + init_betas[i])
        d_r_x    = np.cos(init_pose[2] + init_betas[i])
        d_r_y    = np.sin(init_pose[2] + init_betas[i])
        jacobian = np.array([[d_beta_x, d_r_x],
                            [d_beta_y, d_r_y]])
        jacobian = jacobian[:, :, 0]
        
        # I could have sampled for the measurements as well but this seemed more correct
        landmark_cov[2*i:2*i+2, 2*i:2*i+2] = jacobian @ init_measure_cov @ jacobian.T


    return k, landmark, landmark_cov

def predict(X, P, control, control_cov, k):
    '''
    Predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # Extract out the controls
    dt      = control[0]
    alpha_t = control[1]

    # Update the mean by the control input
    X_out     = X.copy()
    X_out[0]  += dt * np.cos(X[2])
    X_out[1]  += dt * np.sin(X[2])
    X_out[2]  += alpha_t

    # Update the covariance with the jacobians (G_{t+1}) and the process noise (R_{t+1})
    jacobian = np.eye(3 + 2*k) # Make identity so we dont squash the landmark variance
    # This is for the x component of the pose and the landmarks
    jacobian[0, 2]    = -dt * np.sin(X[2])
    # This is for the y component of the pose and the landmarks
    jacobian[1, 2]    = dt * np.cos(X[2])
    
    # Take the control coviariance and apply it to the pose and landmarks
    rot_robot     = np.array([[np.cos(X[2]), -np.sin(X[2]), 0],
                             [np.sin(X[2]), np.cos(X[2]), 0],
                             [0, 0, 1]])
    
    # Rotate the robot to align with its orientation via a matrix
    R      = rot_robot @ control_cov @ rot_robot.T
    R_rsz  = np.zeros((3+2*k, 3+2*k))
    R_rsz[0:3, 0:3]  = R  

    # Apply 
    P_ret = jacobian @ P @ jacobian.T + R_rsz


    return X_out, P_ret

def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    Update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # Extract out the bearing and ranges
    betas  = measure[0::2]
    ranges = measure[1::2]

    # Get landmarks
    l_x = X_pre[0] + ranges * np.cos(X_pre[2] + betas)
    l_y = X_pre[1] + ranges * np.sin(X_pre[2] + betas)

    # Convenience variables
    landmark_measurements = np.hstack((l_x, l_y))
    robot_loc             = np.reshape(np.concatenate((X_pre[0], X_pre[1])), (1, 2))
    robot_loc             = np.repeat(robot_loc, k, axis = 0)

    # Variables to be used in the jacobians and calculations
    delta_x = landmark_measurements[:, 0] - robot_loc[:, 0]
    delta_y = landmark_measurements[:, 1] - robot_loc[:, 1]
    delta   = delta_x**2 + delta_y**2

    # Find the distance to the landmarks
    r = delta**0.5

    # Find the bearing of the landmarks from the robot
    world_angle   = np.arctan2(delta_y, delta_x)
    bearing_angle = world_angle - X_pre[2]
    bearing_angle = wrap2pi(bearing_angle) # Get the angle in the valid range

    Q = np.zeros((2*k, 2*k))

    # Calculate the measurement jacobian (transforms poses to measurements)
    # Measurement x State
    H = np.zeros((2*k, 2*k + 3))
    for i in range(k):
        # Poses
        H[2*i, 0]     = delta_y[i]/delta[i]
        H[2*i, 1]     = -delta_x[i]/delta[i]
        H[2*i, 2]     = -1
        H[2*i+1,   0] = -delta_x[i]/delta[i]**0.5
        H[2*i+1,   1] = -delta_y[i]/delta[i]**0.5

        # Calculate for the measurement of the landmark
        landmark_idx = 3 + 2*i
        H[2*i,         landmark_idx]     = -delta_y[i]/delta[i]
        H[2*i,         landmark_idx + 1] = delta_x[i]/delta[i]
        H[2*i + 1,     landmark_idx]     = delta_x[i]/(delta[i]**0.5)
        H[2*i + 1,     landmark_idx + 1] = delta_y[i]/(delta[i]**0.5)

        # Put in the noise
        Q[2*i:2*i+2, 2*i:2*i+2] = measure_cov

    # Calculate the kalman gain to help form our weighted sum
    K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + Q)

    # Calculate the residual between the empirical measurements and the expected measurements
    h_ut          = np.zeros_like(measure)
    h_ut[0::2, 0] = bearing_angle
    h_ut[1::2, 0] = r
    residual   = measure - h_ut

    # Calculate the mean
    X_pre += K @ residual

    # Calculate the covariance
    P_pre = (np.eye(2*k + 3) - K @ H) @ P_pre

    return X_pre, P_pre


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)

    # Mahalanobis distance
    residual = l_true - X[3:, 0]
    S        = P[3:, 3:]
    residual    = np.reshape(residual, (-1, 1))

    # For loop for the distance
    mah_dists = []
    for i in range(0, len(residual), 2):
        dist  = residual[i:i+2]
        scale = S[i:i+2, i:i+2]
        scaled_dist = dist.T @ scale @ dist
        mah_dists.append(scaled_dist[0, 0])
    mah_dists = np.array(mah_dists)

    # Euclidian Distance
    l_true = np.reshape(l_true, (len(l_true)//2, 2))
    l_pred = np.reshape(X[3:], l_true.shape)
    euclidian   = np.sum((l_true - l_pred)**2, axis = 1)**0.5

    # Print out the P matrix
    print(f"P Matrix: {P}")
    print(f"Euclidian Distances for Landmarks: {euclidian}")
    print(f"Mahalanobis Distance for Landmarks: {mah_dists}")



def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25;
    sig_y = 0.1;
    sig_alpha = 0.1;
    sig_beta = 0.01;
    sig_r = 0.5#sig_r = 0.08;


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("hw2/data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # Initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # Predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # Update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
