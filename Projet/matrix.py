class Matrix():

    def __init__(self, lines, columns, *arg, type = None):

        self.matrix = []
        self.lines = lines
        self.columns = columns

        s = len(arg)

        match type:
            case None:
                if s == lines * columns:
                    for n in range(lines):
                        self.matrix.append(list(arg[n*columns:columns+(n*columns)]))
            case "diag":
                if s == lines and s == columns:
                    for i in range(s):
                        for j in range(s):
                            self.matrix.append(arg[i] if j == i else 0)


    def __str__(self):
        result = ""
        for i in range(self.lines):
            end = "\n" if i != self.lines else ""
            result += str(self.matrix[i]) + end
        return result

    def __getitem__(self, item):
        return self.matrix[item]

    def __mul__(self, other):
        if self.columns == other.lines:
            rlines = self.columns
            rcolumns = other.lines
            result = []
            for i in range(self.lines):
                for j in range(self.columns):
                    c = 0
                    for k in range(self.lines):
                        c += self.matrix[k][j] * other[i][k]
                    result.append(c)
            return Matrix(rlines, rcolumns, *result)