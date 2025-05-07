import pygame
import numpy as np

# Initialiser pygame
pygame.init()

# Dimensions de la fenêtre
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rotation Cube avec la Souris")

class Transform:
    """ Classe pour gérer la transformation d'un objet 3D (position, rotation, échelle) """
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
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

        # Ordre : Échelle, Rotation, puis Translation
        return T @ R @ S

# Définition des rotations
def rotation_x(theta):
    theta = np.radians(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(psi):
    psi = np.radians(psi)
    return np.array([
        [np.cos(psi), 0, np.sin(psi), 0],
        [0, 1, 0, 0],
        [-np.sin(psi), 0, np.cos(psi), 0],
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
    return rotation_z(roll) @ rotation_x(pitch) @ rotation_y(yaw)

def project_vertex(point, d=500):
    """ Projection en 2D avec perspective """
    x, y, z = point
    if z <= 0:
        return None  # Éviter les divisions par zéro
    factor = d / (d + z)
    return np.array([x * factor, y * factor])

def project_vertices(vertices, d=500):
    projected = [project_vertex(v, d) for v in vertices]
    return [None if v is None else (int(v[0] + width // 2), int(v[1] + height // 2)) for v in projected]

# Algorithme de tracé de segment de Bresenham
def bresenham(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def draw_edges(edges, vertices_2d):
    for edge in edges:
        p1, p2 = vertices_2d[edge[0]], vertices_2d[edge[1]]
        if p1 is not None and p2 is not None:
            points = bresenham(p1[0], p1[1], p2[0], p2[1])
            for point in points:
                screen.set_at(point, (0, 0, 0))

# Sommets du cube
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Face arrière
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Face avant
]) * 100  # Agrandir le cube

# Arêtes du cube
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Face arrière
    (4, 5), (5, 6), (6, 7), (7, 4),  # Face avant
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connexions entre faces
]

# Créer un Transform pour le cube
cube_transform = Transform((0, 0, 400),(0, 0, 0),(1, 1, 1))

# Variables de rotation
yaw, pitch, roll = 0, 0, 0

# Boucle principale
running = True
while running:
    screen.fill((255, 255, 255))  # Fond blanc

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            dx, dy = event.rel
            yaw -= dx * 0.3
            pitch += dy * 0.3

    # Mettre à jour la rotation du Transform
    cube_transform.rotation = (yaw, pitch, roll)

    # Calculer la transformation complète
    transform_matrix = cube_transform.get_matrix()

    # Ajouter une colonne de 1 pour utiliser des coordonnées homogènes
    model_vertices = np.hstack((cube_vertices, np.ones((len(cube_vertices), 1))))

    # Appliquer la transformation
    transformed_vertices = (transform_matrix @ model_vertices.T).T[:, :3]

    # Projection des sommets
    projected_vertices = project_vertices(transformed_vertices)

    # Dessiner le cube
    draw_edges(cube_edges, projected_vertices)

    pygame.display.flip()

pygame.quit()
