from abc import ABC, abstractmethod
import pygame
import numpy as np

from game.matrix_utils import generate_movement_axes_from_colmap, get_rotation_about_axis


class Camera(ABC):
    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def update(self, keyspressed, timedelta):
        pass

    @abstractmethod
    def get_modelview():
        pass

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


class PlayerCamera(Camera):

    def __init__(self, colmap_path=None):
        # Model matrix used to align axes with those of the street

        if colmap_path is not None:
            self.up_vector, self.forward_vector, self.side_vector, self.alignment_rotation = generate_movement_axes_from_colmap(colmap_path)
        else:
            self.alignment_rotation = np.identity(4)
            self.up_vector = np.array([0,1,0,1])
            self.forward_vector = np.array([0,0,1,1])
            self.side_vector = np.array([1,0,0,1])

        translation_matrix = np.array([[1,0,0,-0.285],[0,1,0,-0.059],[0,0,1,-0.5718],[0,0,0,1]]).astype(np.float32)
        self.camera_translation = translation_matrix

        # Align camera rotation with the forward_pointing vector
        self.camera_pitch_rotation = np.identity(4)
        self.camera_yaw_rotation = np.identity(4)
        self.camera_roll_rotation = np.identity(4)
        print(f"UP VECTOR: {self.up_vector}")
        print(f"SIDE VECTOR: {self.side_vector}")
        print(f"FORWARDS VECTOR: {self.forward_vector}")

    def _translate_camera(self, direction, timedelta, f=0.001):
        self.camera_translation[0][3] += direction[0] * timedelta * f
        self.camera_translation[1][3] += direction[1] * timedelta * f
        self.camera_translation[2][3] += direction[2] * timedelta * f

    def _update_pointing(self, theta):
        rotation = get_rotation_about_axis(theta, self.up_vector)
        self.forward_vector = rotation @ self.forward_vector
        self.side_vector = rotation @ self.side_vector

    def update(self, keyspressed, timedelta):
        # Translation forwards/backwards
        if keyspressed[pygame.K_w]:
            self._translate_camera(-self.forward_vector, timedelta)
        elif keyspressed[pygame.K_s]:
            self._translate_camera(self.forward_vector, timedelta)

        # Translation sideways
        if keyspressed[pygame.K_a]:
            self._translate_camera(self.side_vector, timedelta)
        elif keyspressed[pygame.K_d]:
            self._translate_camera(-self.side_vector, timedelta)

        # Translation up/down
        if keyspressed[pygame.K_q]:
            self._translate_camera(self.up_vector, timedelta)
        elif keyspressed[pygame.K_e]:
            self._translate_camera(-self.up_vector, timedelta)

        # camera rotations
        if keyspressed[pygame.K_i]:
            self.camera_pitch_rotation = self.camera_pitch_rotation @ self._rotate_z(-timedelta/1500)
        elif keyspressed[pygame.K_k]:
            self.camera_pitch_rotation = self.camera_pitch_rotation @ self._rotate_z(timedelta/1500)
        if keyspressed[pygame.K_j]:
            self.camera_yaw_rotation = self.camera_yaw_rotation @ self._rotate_y(-timedelta/1500)
            self._update_pointing(timedelta/1500)
        elif keyspressed[pygame.K_l]:
            self.camera_yaw_rotation = self.camera_yaw_rotation @ self._rotate_y(timedelta/1500)
            self._update_pointing(-timedelta/1500)
        if keyspressed[pygame.K_u]:
            self.camera_roll_rotation = self.camera_roll_rotation @ self._rotate_x(timedelta/1500)
        elif keyspressed[pygame.K_o]:
            self.camera_roll_rotation = self.camera_roll_rotation @ self._rotate_x(-timedelta/1500)
        
    def get_modelview(self):
        camera_rotation = self.alignment_rotation @ self.camera_roll_rotation @ self.camera_pitch_rotation @ self.camera_yaw_rotation
        return camera_rotation @ self.camera_translation @ np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])


class DevCamera(Camera):
    def __init__(self):
        self.translation_matrix = np.identity(4)
        self.xrot = np.identity(4)
        self.yrot = np.identity(4)
        self.zrot = np.identity(4)

    def update(self, keyspressed, timedelta):
        # Rotations about the origin
        if keyspressed[pygame.K_i]:
            self.xrot = self.xrot @ self._rotate_x(timedelta/1500)
        if keyspressed[pygame.K_k]:
            self.xrot = self.xrot @ self._rotate_x(-timedelta/1500)
        if keyspressed[pygame.K_j]:
            self.yrot = self.yrot @ self._rotate_y(timedelta/1500)
        if keyspressed[pygame.K_l]:
            self.yrot = self.yrot @ self._rotate_y(-timedelta/1500)
        if keyspressed[pygame.K_u]:
            self.zrot = self.zrot @ self._rotate_z(timedelta/1500)
        if keyspressed[pygame.K_o]:
            self.zrot = self.zrot @ self._rotate_z(-timedelta/1500)

        # Translations towards or away from the origin
        if keyspressed[pygame.K_w]:
            self.translation_matrix += np.array([[0,0,0,0],[0,0,0,0],[0,0,0,timedelta/1500],[0,0,0,0]])
        if keyspressed[pygame.K_s]:
            self.translation_matrix += np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-timedelta/1500],[0,0,0,0]])

    def get_modelview(self):
        return self.translation_matrix @ self.zrot @ self.yrot @ self.xrot
