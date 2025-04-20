import numpy as np

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def rotation_x(theta):
    theta = np.radians(theta)  # Convertir en radians
    return np.array([
        [1, 0, 0, 0],
        [0, cos(theta), -sin(theta), 0],
        [0, sin(theta), cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(psi):
    psi = np.radians(psi)
    return np.array([
        [cos(psi), 0, sin(psi), 0],
        [0, 1, 0, 0],
        [-sin(psi), 0, cos(psi), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(phi):
    phi = np.radians(phi)
    return np.array([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix(yaw, pitch, roll):
    R_x = rotation_x(pitch)
    R_y = rotation_y(yaw)
    R_z = rotation_z(roll)

    return R_z @ R_x @ R_y  # Ordre ZXY (modifiable)

# Matrice d'échelle
def scale_matrix(scale):
    return np.diag([scale[0], scale[1], scale[2], 1])

def translation_martrix(translation):
    T = np.eye(4)
    T[:3, 3] = translation  # Déplace le modèle
    return T


def transformation_matrix(translation, rotation = (0,0,0), scale = (1,1,1)):
    # Matrice d'échelle
    S = scale_matrix(scale)

    # Matrice de rotation (exemple : uniquement sur Y)
    R = rotation_matrix(rotation[0],rotation[1],rotation[2])

    # Matrice de translation
    T = translation_martrix(translation)

    # Multiplication des matrices : T * R * S
    return T @ R @ S

# Appliquer la transformation
model_matrix = transformation_matrix([0, 0, -5],(0,0,0), (1,1,1))  # Position, Rotation, Échelle