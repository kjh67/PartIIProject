import pygame
import numpy as np


class PlayerCamera():

    def __init__(self, model_matrix=np.identity(4)):
        # Model matrix used to align axes with those of the street
        self.model_rotation = model_matrix
        self.camera_translation = np.identity(4)
        self.camera_rotation = np.identity(4)

        self.forward_direction = np.array([0,0,1,1])
        self.sideways_direction = np.array([1,0,0,1])
        self.updown_direction = np.array([0,1,0,1])

        self.pitch = 0
        self.yaw = 0
        self.roll = 0

    def _translate_camera(self, direction, timedelta, f=0.001):
        self.camera_translation[0][3] += direction[0] * timedelta * f
        self.camera_translation[1][3] += direction[1] * timedelta * f
        self.camera_translation[2][3] += direction[2] * timedelta * f

    def update(self, keyspressed, timedelta):
        # Translation forwards/backwards
        if keyspressed[pygame.K_w]:
            self._translate_camera(self.forward_direction, timedelta)
        elif keyspressed[pygame.K_s]:
            self._translate_camera(-self.forward_direction, timedelta)

        # Translation sideways
        if keyspressed[pygame.K_a]:
            self._translate_camera(-self.sideways_direction, timedelta)
        elif keyspressed[pygame.K_d]:
            self._translate_camera(self.sideways_direction, timedelta)

        # Translation up/down
        if keyspressed[pygame.K_q]:
            self._translate_camera(self.updown_direction, timedelta)
        elif keyspressed[pygame.K_e]:
            self._translate_camera(-self.updown_direction, timedelta)
        

        # Camera rotations; pitch, yaw, roll
        # Yaw, rotation in y axis; affects pointing vectors
        pointing_adjustment = np.identity(4)
        if keyspressed[pygame.K_j]:
            self.yaw += timedelta/1500
            pointing_adjustment = self._rotate_y(-timedelta/1500)
        elif keyspressed[pygame.K_l]:
            self.yaw -= timedelta/1500
            pointing_adjustment = self._rotate_y(timedelta/1500)
        # Update pointing vectors
        self.forward_direction = pointing_adjustment @ self.forward_direction
        self.sideways_direction = pointing_adjustment @ self.sideways_direction
    
        # Pitch, rotation in x axis; does not affect pointing vectors
        if keyspressed[pygame.K_i]:
            self.pitch += timedelta / 1500
        elif keyspressed[pygame.K_k]:
            self.pitch -= timedelta / 1500

        # Roll, rotation in z axis; does not affect pointing vectors
        if keyspressed[pygame.K_u]:
            self.roll += timedelta / 1500
        elif keyspressed[pygame.K_o]:
            self.roll -= timedelta / 1500

        # Apply rotations in a consistent order; yaw, then pitch, then roll
        self.camera_rotation = self._rotate_z(self.roll)\
                               @ self._rotate_x(self.pitch)\
                               @ self._rotate_y(self.yaw)


    def get_modelview(self):
        return  self.camera_rotation @ self.camera_translation @ self.model_rotation


    def _rotate_x(self, theta):
        return np.array([[1,0,0,0],
                        [0,np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0,0,0,1]])

    def _rotate_y(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                        [0,1,0,0],
                        [-np.sin(theta), 0, np.cos(theta), 0],
                        [0,0,0,1]])

    def _rotate_z(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0,0,1,0],
                        [0,0,0,1]])
