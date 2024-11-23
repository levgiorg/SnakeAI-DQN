class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Implement equality for easy comparison
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # For printing the point
    def __repr__(self):
        return f'Point({self.x}, {self.y})'
