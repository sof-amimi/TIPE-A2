import pygame
import numpy as np

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Python Renderer")

# Definition des matrices de rotations selon les axes x, y et z

def rotation_x(theta : float):
    """
    Matrice de rotation theta selon l'axe x = (1,0,0)
    :param theta: Angle en degrée
    :return: Matrice de rotation selon l'axe x de l'angle theta
    """
    theta = np.radians(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(psi : float):
    """
    Matrice de rotation psi selon l'axe y = (0,1,0)
    :param psi: Angle en degrée
    :return: Matrice de rotation selon l'axe y de l'angle psi
    """
    psi = np.radians(psi)
    return np.array([
        [np.cos(psi), 0, np.sin(psi), 0],
        [0, 1, 0, 0],
        [-np.sin(psi), 0, np.cos(psi), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(phi : float):
    """
    Matrice de rotation phi selon l'axe z = (0,0,1)
    :param phi: Angle en degrée
    :return: Matrice de rotation selon l'axe z de l'angle phi
    """
    phi = np.radians(phi)
    return np.array([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Calcul d'une rotation selon les 3 axes en simultané par la multiplication des matrices de rotations selon leurs axes repsectifs

def rotation_matrix(yaw, pitch, roll):
    return rotation_z(roll) @ rotation_x(pitch) @ rotation_y(yaw)


class Transform:

    def __init__(self,position : tuple, rotation : tuple, scale : tuple):
        """
        :param position: Tuple de 3 float représentant une position dans l'espace par rapport au point (0,0,0)
        :param rotation: Tuple de 3 float représentant la rotation selon les axes x, y et z
        :param scale: Tuple de 3 float représentant la mise à l'echelle sur les axes x, y et z
        """
        self.position = np.array(position)
        self.rotation = np.array(rotation)
        self.scale = np.array(scale)

    def get_matrix(self):
        """ Génère la matrice de transformation 4x4 """
        # Matrice d'échelle
        S = np.diag([*self.scale, 1])

        # Matrice de rotation
        R = rotation_matrix(*self.rotation)

        # Matrice de translation
        T = np.eye(4)
        T[:3, 3] = self.position  # Ajouter la translation

        return T @ R @ S

def projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])
