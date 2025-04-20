from math import cos, sin

class Matrix4:
    def __init__(self, values=None):
        if values:
            self.values = values
        else:
            self.values = [[1 if i == j else 0 for i in range(4)] for j in range(4)]

    def __mul__(self, other):
        if isinstance(other, Matrix4):
            result = [[sum(self.values[i][k] * other.values[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
            return Matrix4(result)
        elif isinstance(other, list):  # Multiplication par un vecteur (coordonnées homogènes)
            return [sum(self.values[i][j] * other[j] for j in range(4)) for i in range(4)]

    @staticmethod
    def identity():
        return Matrix4()

    @staticmethod
    def translation(tx, ty, tz):
        mat = Matrix4.identity()
        mat.values[0][3] = tx
        mat.values[1][3] = ty
        mat.values[2][3] = tz
        return mat

    @staticmethod
    def scale(sx, sy, sz):
        mat = Matrix4.identity()
        mat.values[0][0] = sx
        mat.values[1][1] = sy
        mat.values[2][2] = sz
        return mat

    @staticmethod
    def rotation_x(theta):
        theta = theta * 3.14159265 / 180  # Conversion en radians
        mat = Matrix4.identity()
        mat.values[1][1] = cos(theta)
        mat.values[1][2] = -sin(theta)
        mat.values[2][1] = sin(theta)
        mat.values[2][2] = cos(theta)
        return mat

    @staticmethod
    def rotation_y(psi):
        psi = psi * 3.14159265 / 180
        mat = Matrix4.identity()
        mat.values[0][0] = cos(psi)
        mat.values[0][2] = sin(psi)
        mat.values[2][0] = -sin(psi)
        mat.values[2][2] = cos(psi)
        return mat

    @staticmethod
    def rotation_z(phi):
        phi = phi * 3.14159265 / 180
        mat = Matrix4.identity()
        mat.values[0][0] = cos(phi)
        mat.values[0][1] = -sin(phi)
        mat.values[1][0] = sin(phi)
        mat.values[1][1] = cos(phi)
        return mat

    @staticmethod
    def rotation(yaw, pitch, roll):
        return Matrix4.rotation_z(roll) * Matrix4.rotation_x(pitch) * Matrix4.rotation_y(yaw)

class Transform:
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def get_matrix(self):
        S = Matrix4.scale(*self.scale)
        R = Matrix4.rotation(*self.rotation)
        T = Matrix4.translation(*self.position)
        return T * R * S
