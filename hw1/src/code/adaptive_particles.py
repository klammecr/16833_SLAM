import numpy as np

class AdaptiveParticleCalculator:
    def __init__(self, num_initial, min_particles = 10):
        self.num_initial   = num_initial
        self.min_particles = min_particles
        self.prev_entropy  = None

    def calculate_naive(self, X_bar):
        num_particles = X_bar.shape[0]
        probs         = X_bar[:, -1]
        probs /= np.sum(probs)
        entropy       = -probs * np.log(probs)

        # Only update number of particles if we have same # of particles as last time and we are not in ts 1
        if self.prev_entropy is not None and self.prev_entropy.shape[0] == num_particles:
            num_particles *= np.exp(10000*(np.sum(entropy) - np.sum(self.prev_entropy)))
            num_particles =  max(round(num_particles), self.min_particles)

        # Set previous entropy
        self.prev_entropy = entropy

        sort_idxs = np.argsort(probs)[:num_particles]
        X_bar = X_bar[sort_idxs]

        return X_bar, sort_idxs
