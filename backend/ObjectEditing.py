import numpy as np

class MultyDimentionObject:
    def __init__(self, txt_filename):
        self.points = None
        self.read_points_from_file(txt_filename)

    def read_points_from_file(self, filename):
        points = []
        with open(filename, 'r') as f:
            for line in f:
                x, y, z = map(float, line.split()[0:3])
                points.append([x, y, z])
        self.points = np.array(points)

        return self

    def rotate_points(self, angle_deg, axis='z'):
        angle = np.radians(angle_deg)
        
        if axis == 'z':
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle),  np.cos(angle), 0],
                            [0, 0, 1]])
        elif axis == 'y':
            rot = np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == 'x':
            rot = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
        else:
            raise ValueError("Ось должна быть 'x', 'y' или 'z'")
        
        centroid = np.mean(self.points, axis=0)
        shifted = self.points - centroid
        rotated = (rot @ shifted.T).T
        self.points = rotated + centroid
        return self

    def object_scaling(self, target_size):
        current_size = np.ptp(self.points, axis=0).max()

        scale = target_size / current_size

        centroid = np.mean(self.points, axis=0)
        
        self.points = (self.points - centroid) * scale

        return self

    def one_axis_move(self, step, axis='z'):

        if axis == 'x':
            self.points[:, 0] += step

        elif axis == 'y':
            self.points[:, 1] += step

        elif axis == 'z':
            self.points[:, 2] += step

        else:
            raise ValueError("Ось должна быть 'x', 'y' или 'z'")
        
        return self

    def replace(self, target_place):
    
        centroid = np.mean(self.points, axis=0)
        self.points = self.points - centroid + target_place
    
        return self