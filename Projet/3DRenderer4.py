import pygame
import pyassimp
import numpy as np

# Initialiser pygame
pygame.init()

# Dimensions de la fenêtre
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Python Renderer")


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

def projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / np.tan(np.radians(fov) / 2)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])



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


class Camera:
    """ Représente une caméra dans l'espace 3D """
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0)):
        self.transform = Transform(position, rotation)  # Utilise Transform (Le scale par default est (1,1,1))

    def get_view_matrix(self):
        """ Retourne la matrice de vue 4x4 """
        # Prendre l'inverse de la transformation de la caméra
        R_inv = rotation_matrix(-self.transform.rotation[0],
                                -self.transform.rotation[1],
                                -self.transform.rotation[2])

        T_inv = np.eye(4)
        T_inv[:3, 3] = -self.transform.position  # Déplace le monde en sens inverse

        return R_inv @ T_inv  # Rotation puis translation



def project_vertices(vertices, p):
    projected = []
    for v in vertices:
        vec = np.append(v, 1)  # Passer en coordonnées homogènes (Vecteur 3x1 -> Vecteur 4x1)
        transformed = p @ vec

        # Perspective divide (division par W)
        if transformed[3] != 0:
            transformed /= transformed[3]

        # Convertir en coordonnées écran
        x_screen = int((transformed[0] + 1) * width / 2)
        y_screen = int((1 - transformed[1]) * height / 2)

        projected.append((x_screen, y_screen))
    return projected


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


def get_edges_from_faces(mesh):
    edges = set()
    for face in mesh.faces:
        if len(face) >= 2:
            for i in range(len(face)):
                a = face[i]
                b = face[(i + 1) % len(face)]
                edge = tuple(sorted((a, b)))
                edges.add(edge)
    return list(edges)


def parcourir_nodes(mdl, node, mesh_data, parent_transform = (0,0,0)):
    local_transform = node.transformation.astype(np.float64)
    global_transform = parent_transform @ local_transform

    for mesh_index in node.meshes:
        mesh = mdl.meshes[mesh_index]
        vertices = mesh.vertices

        for v in vertices:
            v_homo = np.append(v, 1)
            transformed = global_transform @ v_homo
            transformed_vertices(transformed[:3])

        transformed_edge = get_edges_from_faces(mesh)

        mesh_data.append({
            "vertices": np.array(transformed_vertices),
            "edges": np.array(transformed_edge)
        })

    for child in node.children:
        parcourir_nodes(child, global_transform, mdl, mesh_data)


def load_mesh_data(m):
    mdl = assimp.load(m)
    root = mdl.rootnode
    mesh_data = []
    parcourir_nodes(mdl,root,mesh_data)
    return mesh_data


# Sommets du cube (centré et agrandi pour une meilleure visibilité)
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Face arrière
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Face avant
    ])

# Arêtes du cube
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Face arrière
    (4, 5), (5, 6), (6, 7), (7, 4),  # Face avant
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connexions entre faces
]

# Variables de rotation
yaw, pitch, roll = 0, 0, 0

# Matrice origine monde
origin = Transform()

mesh_data = [[cube_vertices, cube_edges]]
model_transform = Transform((0,0,5),(0,0,0),(1,1,1))
proj_matrix = projection_matrix(90, width / height, 0.1, 1000)


mesh_data = load_mesh_data("Couch.obj")


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
            pitch -= dy * 0.3

    for mesh in mesh_data:
        # Ajouter une colonne de 1 pour utiliser des coordonnées homogènes (matrice 4x4)
        model_vertices = np.hstack((mesh[0], np.ones((len(mesh[0]), 1))))

        # Appliquer la rotation et la projection
        rotated_vertices = np.dot(model_vertices, rotation_matrix(yaw, pitch, roll))

        # Appliquer la transformation globale
        transformed_vertices = (model_transform.get_matrix() @ rotated_vertices.T).T[:, :3]

        projected_vertices = project_vertices(transformed_vertices, proj_matrix)


        # Dessiner le model
        draw_edges(mesh[1], projected_vertices)

    pygame.display.flip()

pygame.quit()
