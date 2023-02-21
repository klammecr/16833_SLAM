'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        pass

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        X_bar_resampled =  np.zeros_like(X_bar)

        # X_bar contains the weight (probability of resampling) for all particles
        samples = np.random.multinomial(X_bar.shape[0], X_bar[:, 3])

        for idx, sample in enumerate(samples):
            X_bar_resampled[idx] = X_bar[sample]

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        Outcomes:

        - Select samples independently of one another
        - We want to make sure that when we have many of the same particles, we don't destroy the variance as an estimator
        - We want to make sure we want to not lose diversity and just create samples of the same particle over and over again

        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        # Resampled particles
        X_bar_resampled =  np.zeros_like(X_bar)

        # Init variables
        M      = X_bar.shape[0]
        r      = np.random.uniform(0, 1.0/M) 
        w      = X_bar[:, 3]
        w     /= np.sum(w)
        c      = w[0]
        i      = 0

        # DEBUG:
        # self.visualize_low_variance_resampler(U_list, X_bar)

        # Here we will go through the particles and start the interplay of c and U
        # U will increment a fixed amount each iteration
        # c will essentially cumulatively keep track of the sum of the particles weights so far and will look to exceed U
        # Therefore, Very small weighted particles are unlikely but could possibly be sampled, higher weights have a proportionally better chance
        for m in range(M):
            # Upper bound, when c surpasses this threshold, sample the particle
            U = r + m * (1.0/M)

            while U > c:
                i += 1
                c += w[i]

            # c caught up to U, sample the particle
            X_bar_resampled[m] = X_bar[i]

        return X_bar_resampled

    def visualize_low_variance_resampler(self, U_list, particles):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # set up the figure
        h = .25
        fig = plt.figure(3)
        ax = fig.add_subplot(111)
        ax.set_xlim(-0.05,1.05)
        ax.set_ylim(-h, h)
        plt.hlines(0, 0, 1)

        # Plot the particle cumulative probs
        particle_probs = np.cumsum(particles[:, -1])

        for idx in range(len(particle_probs)):
            if idx > 0:
                start = particle_probs[idx-1]
            else:
                start = 0
            # Add the rectangle
            ax.add_patch(Rectangle((start, -h/16), particle_probs[idx] - start, h/8, ec ='k'))

        
        # Plot the U_list
        plt.vlines(U_list, -h/8, h/8, colors="r", label="Samples")

        # Extras
        plt.title("Low Variance Resampling Visualization")
        plt.axis('off')
        plt.legend()
        plt.show()
        plt.close()