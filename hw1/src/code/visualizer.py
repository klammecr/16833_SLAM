# Third Party
import cv2
import os
import numpy as np

class Visualizer():
    def __init__(self, occ_map, output_path, name, steps = 10, resolution = 10, video = True):
        # Internals
        self.prev_state    = None
        self.output_path   = output_path
        self.steps         = steps
        self.resolution    = resolution
        self.name          = name

        # Maks to help display rays
        self.ray_mask = np.zeros_like(occ_map)

        # Process occupancy map
        #scale and convert map to 3 channels
        #scale
        occ_map = occ_map.copy()
        occ_map[occ_map < 0] = 0
        occ_map = (occ_map - occ_map.min())/(occ_map.max() - occ_map.min())
        occ_map = (occ_map*255).astype('uint8')
        # Stack the BGR channels
        occ_map = np.stack((occ_map, occ_map, occ_map), axis=-1)
        self.occupancy_map = occ_map

        # Optionally visualize in video format
        self.video_writer = None
        if video:
            self.init_video()

    def __del__(self):
        cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()

    def init_video(self):
        #format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        #resolution of frames
        dim = self.occupancy_map.shape[0]
        
        #video writer instance
        self.video_writer = cv2.VideoWriter(os.path.join(self.output_path), fourcc, 20.0, (dim, dim))

    def step_video(self, map_vis):
        cv2.imshow('Output', map_vis)
        if self.video_writer:
            self.video_writer.write(map_vis)

    def set_ray_mask(self, map):
        self.ray_mask = (map == -5)

    def visualize_ranges(self, robot_pos, z_true):
        self.p1 = None
        self.p2 = None
        if z_true is None:
            return

        # DEBUG THE RANGES
        best_particle = np.argmax(robot_pos[:, -1])

        self.p1 = []
        self.p2 = []
        for ang_idx, ang in enumerate(range(-90, 90, 5)):
            p2_x = robot_pos[best_particle, 0] + z_true[best_particle, ang_idx] * np.cos(robot_pos[best_particle, 2] + ang * np.pi/180)
            p2_y = robot_pos[best_particle, 1] + z_true[best_particle, ang_idx] * np.sin(robot_pos[best_particle, 2] + ang * np.pi/180)
            # Convert from cm to pix
            p1_x = robot_pos[best_particle, 0] // 10
            p1_y = robot_pos[best_particle, 1] // 10
            p2_x /= 10
            p2_y /= 10

            self.p1.append((int(p1_x), int(p1_y)))
            self.p2.append((int(p2_x), int(p2_y)))

    def visualize_timestep(self, X_bar, tstep, sort_idxs):
        #compute coordiantes
        x_locs_pix = X_bar[:, 0] // self.resolution
        y_locs_pix = X_bar[:, 1] // self.resolution
        x_locs_pix = x_locs_pix.astype('int')
        y_locs_pix = y_locs_pix.astype('int')
        
        # Deep copy the map
        occ_map = self.occupancy_map.copy()

        # Particle specifics
        probs    = X_bar[:, -1]
        best_idx = np.argmax(probs)

        # Orientation precomputation
        orientation_rad = X_bar[:, 2]
        x_end = x_locs_pix + self.steps * np.cos(orientation_rad)
        y_end = y_locs_pix + self.steps * np.sin(orientation_rad)

        # Sketch a blue line for the trajectory
        if self.prev_state is not None:
            if sort_idxs is None:
                dx = self.prev_state[0] - x_locs_pix
                dy = self.prev_state[1] - y_locs_pix
            else:
                dx = self.prev_state[0][sort_idxs] - x_locs_pix
                dy = self.prev_state[1][sort_idxs] - y_locs_pix
            for i in range(len(dx)):
                # Only sketch the best particle
                if i == best_idx:
                    # If the particle gets resampled, don't trace it
                    if (abs(dx[i]) <= 2 and abs(dy[i]) <= 2) and (abs(dx[i]) > 0 or abs(dy[i]) > 0):
                        cv2.line(occ_map, (self.prev_state[0][i], self.prev_state[1][i]), (x_locs_pix[i], y_locs_pix[i]), color = (255,0,0), thickness = 1)

        # The trajectory is the only thing that is permanent frame-to-frame, cache it in the class now
        # The rest is for single frame display
        self.occupancy_map = occ_map.copy()

        for idx, xyxy in enumerate(zip(x_locs_pix, y_locs_pix, x_end, y_end)):
            # Get start and end
            x_s, y_s, x_e, y_e = xyxy

            # Add particles to map as red
            if idx == best_idx:
                # Draw a BIG particle
                cv2.circle(occ_map, (x_s, y_s), radius = 3, color = (0,0,255), thickness = 5)

                # Create a green line to visualize the orientation of the robot
                cv2.line(occ_map, (x_s, y_s), (int(x_e), int(y_e)), color = (0, 255, 0), thickness=1)

                # Show the rays for the dominant particle

            else:
                # Draw a very small particle
                cv2.circle(occ_map, (x_s, y_s), radius = 1, color = (0,0,255), thickness = 1)

            # Increment index
            idx += 1
 
        # Create the rays as white
        if self.p1 is not None and self.p2 is not None:
            for i in range(len(self.p1)):
                cv2.line(occ_map, self.p1[i], self.p2[i], color = (255, 255,255))

        # Update prev
        self.prev_state = [x_locs_pix, y_locs_pix]

        # Flip map
        occ_map = cv2.flip(occ_map, 0)

        # Display number of particles
        cv2.putText(occ_map, text = f"{self.name} | Number of Particles: {X_bar.shape[0]}", org=[25,30], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,255,255))

        # Give the occupancy map to the video writer
        self.step_video(occ_map)
        
        return occ_map