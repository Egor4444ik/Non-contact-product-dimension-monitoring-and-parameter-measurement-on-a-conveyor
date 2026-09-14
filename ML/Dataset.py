import numpy as np
import torch
from torch.utils.data import Dataset

def random_rotation_3d(points):
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    psi = np.random.uniform(0, 2*np.pi)
    Rx = np.array([[1,0,0],
                   [0,np.cos(theta),-np.sin(theta)],
                   [0,np.sin(theta),np.cos(theta)]])
    Ry = np.array([[np.cos(phi),0,np.sin(phi)],
                   [0,1,0],
                   [-np.sin(phi),0,np.cos(phi)]])
    Rz = np.array([[np.cos(psi),-np.sin(psi),0],
                   [np.sin(psi),np.cos(psi),0],
                   [0,0,1]])
    R = Rz @ Ry @ Rx
    return points @ R.T

def random_scale(points, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return points * scale

def random_translate(points, shift_range=0.2):
    shift = np.random.uniform(-shift_range, shift_range, size=3)
    return points + shift

def add_noise(points, noise_std=0.02):
    noise = np.random.normal(0, noise_std, size=points.shape)
    return points + noise

def augment_pointcloud(points):
    points = random_rotation_3d(points)
    points = random_scale(points)
    points = random_translate(points)
    points = add_noise(points)
    return points

def sample_points(points, num_points):
    if len(points) >= num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
    else:
        idx = np.random.choice(len(points), num_points, replace=True)
    return points[idx]

def generate_sphere(num_points, radius=30.0):
    theta = np.random.uniform(0, 2*np.pi, size=num_points)
    phi = np.arccos(2*np.random.uniform(0, 1, size=num_points) - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1)
    centroid = np.mean(points, axis=0)
    random_shift = np.random.uniform(-100, 100, size=3)
    points = points - centroid + random_shift

    return points

def generate_cube(num_points, side=30.0):
    points = []
    for _ in range(num_points):
        face = np.random.randint(0, 6)
        if face == 0:
            p = [side/2, np.random.uniform(-side/2, side/2), np.random.uniform(-side/2, side/2)]
        elif face == 1:
            p = [-side/2, np.random.uniform(-side/2, side/2), np.random.uniform(-side/2, side/2)]
        elif face == 2:
            p = [np.random.uniform(-side/2, side/2), side/2, np.random.uniform(-side/2, side/2)]
        elif face == 3:
            p = [np.random.uniform(-side/2, side/2), -side/2, np.random.uniform(-side/2, side/2)]
        elif face == 4:
            p = [np.random.uniform(-side/2, side/2), np.random.uniform(-side/2, side/2), side/2]
        else:
            p = [np.random.uniform(-side/2, side/2), np.random.uniform(-side/2, side/2), -side/2]
        points.append(p)

    points = np.array(points)
    centroid = np.mean(points, axis=0)
    random_shift = np.random.uniform(-100, 100, size=3)
    points = points - centroid + random_shift
    
    return np.array(points)

def generate_random_shape(num_points):
    if np.random.rand() < 0.5:
        return generate_sphere(num_points)
    else:
        return generate_cube(num_points)

class PointCloudDataset(Dataset):
    """
    Датасет для сегментации одного объекта.
    object_path: путь к .txt с координатами (x y z) объекта.
    num_points: количество точек в облаке.
    object_label: метка для точек объекта (по умолчанию 1).
    background_label: метка для фона/шума (0).
    prob_object: вероятность взять аугментированный объект, иначе случайную фигуру.
    """
    def __init__(self, object_path, num_points=2048, object_label=1, background_label=0,
                 prob_object=0.7, augment=True, num_categories=1, category=0):
        self.num_points = num_points
        self.object_label = object_label
        self.background_label = background_label
        self.prob_object = prob_object
        self.augment = augment
        self.num_categories = num_categories
        self.category = category

        self.object_points = np.loadtxt(object_path)
        centroid = np.mean(self.object_points, axis=0)
        self.object_points = self.object_points - centroid

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        r = np.random.rand()

        if r < self.prob_object / 5:
            pts = sample_points(self.object_points, self.num_points)
            if self.augment:
                pts = augment_pointcloud(pts)
            labels = np.full(self.num_points, self.object_label, dtype=np.int64)

        elif r < 4 * self.prob_object / 5:
            bg_ratio = np.random.uniform(0.1, 1.1)

            num_bg = int(round(self.num_points * bg_ratio / (1.0 + bg_ratio)))
            num_obj = self.num_points - num_bg

            obj_pts = sample_points(self.object_points, num_obj)
            bg_pts = generate_random_shape(num_bg)

            pts = np.vstack([obj_pts, bg_pts])
            labels = np.concatenate([
                np.full(num_obj, self.object_label, dtype=np.int64),
                np.full(num_bg, self.background_label, dtype=np.int64)
            ])

            shuffle_idx = np.random.permutation(self.num_points)
            pts = pts[shuffle_idx]
            labels = labels[shuffle_idx]

            if self.augment:
                pts = augment_pointcloud(pts)

        else:
            pts = generate_random_shape(self.num_points)
            if self.augment:
                pts = augment_pointcloud(pts)
            labels = np.full(self.num_points, self.background_label, dtype=np.int64)

        pts = torch.from_numpy(pts).float()
        labels = torch.from_numpy(labels).long()
        cat = torch.tensor(self.category, dtype=torch.long)
        return pts, labels, cat