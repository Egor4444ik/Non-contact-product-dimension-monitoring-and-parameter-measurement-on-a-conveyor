import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import pyvista

def save_points(points, filename):
    with open(filename, 'w') as f:
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

def read_points_from_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z = map(float, line.split()[0:3])
            points.append([x, y, z])
    return np.array(points)

def rotate_points(points, angle_deg, axis='z'):
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
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    rotated = (rot @ shifted.T).T
    return rotated + centroid

def are_coplanar(points, tol=1e-8):
    if points.shape[0] < 3:
        return True
    p0 = points[0]
    p1 = points[1]
    normal = None
    for i in range(2, points.shape[0]):
        normal = np.cross(p1 - p0, points[i] - p0)
        if np.linalg.norm(normal) > tol:
            break
    else:
        return True
    normal = normal / np.linalg.norm(normal)
    for pt in points:
        if abs(np.dot(pt - p0, normal)) > tol:
            return False
    return True

def compute_volume_convex(points):
    if points.shape[0] < 4:
        raise ValueError("Нужно минимум 4 точки.")
    if are_coplanar(points):
        return 0.0, None
    hull = ConvexHull(points)
    return hull.volume, hull

def render_convex_hull(points, hull, title="Выпуклая оболочка"):
    """Отображает точки и грани выпуклой оболочки (только исходный объект)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='blue', s=10, alpha=0.6, label='Точки')
    if hull is not None:
        faces = hull.simplices
        mesh = Poly3DCollection(points[faces], alpha=0.25, facecolor='cyan',
                                edgecolor='k', linewidths=0.5)
        ax.add_collection3d(mesh)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid = np.mean(points, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.legend()
    plt.show()

def load_product(target_center, target_size):
    """
    Загружает чайник, масштабирует его так, чтобы его размах стал равен target_size,
    и помещает центром в target_center.
    """
    product_filename = "product.txt"
    pts = read_points_from_file(product_filename)  # массив (N,3)
    if pts.shape[0] == 0:
        return None

    current_size = np.ptp(pts, axis=0).max()
    if current_size == 0:
        return None

    scale = target_size / current_size

    centroid = np.mean(pts, axis=0)
    x = (pts[0] - centroid[0]) * scale + target_center[0]
    y = (pts[1] - centroid[1]) * scale + target_center[1]
    z = (pts[2] - centroid[2]) * scale + target_center[2]
    scaled = (pts - centroid) * scale

    min_z = scaled[:, 2].min()
    scaled[:, 0] += target_center[0]
    scaled[:, 1] += target_center[1]
    scaled[:, 2] += target_center[2] - min_z

    return scaled

def visualize_with_product(points, product_points, title="Объект + чайник"):
    plotter = pyvista.Plotter(window_size=(1000, 700))
    plotter.add_points(points, color='blue', point_size=3, label='Объект (точки)')
    plotter.add_points(product_points, color='red', point_size=5, label='Чайник (точки)')
    plotter.add_legend()
    plotter.add_text(title, font_size=14)
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show(interactive=True)

def visualize_1_obj(points, title="1 объект"):
    plotter = pyvista.Plotter(window_size=(1000, 700))
    plotter.add_points(points, color='blue', point_size=3, label='Объект (точки)')
    plotter.add_legend()
    plotter.add_text(title, font_size=14)
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show(interactive=True)

if __name__ == "__main__":
    filename = "barbell.txt"#"products_production/frame_0060.txt"#"ConveyorBeltAssemblyLine.txt"#"product.txt"#

    points = read_points_from_file(filename)
    points = rotate_points(points, 270, axis='x')
    if points.shape[0] == 0:
        print("Не удалось прочитать ни одной точки.")
        exit()

    print(f"Загружено точек: {points.shape[0]}")
    
    volume, hull = compute_volume_convex(points)
    
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    max_z = np.max(points[:, 2])
    target_point = np.array([center_x, center_y, max_z])

    bbox_size = np.ptp(points, axis=0).max()
    radius = bbox_size * 0.05               
    print(f"Размер объекта: {bbox_size:.2f}, радиус чайника: {radius:.2f}")

    visualize_1_obj(points)

    """target_size = np.ptp(points, axis=0).max()/10
    product = load_product(target_point, target_size)
    product = rotate_points(product, 90, axis='x')

    if product is None:
        print("Чайник не загружен. Выход.")
        exit()

    all_points = np.vstack([product, points])
    save_points(all_points, "productOnConveyor.txt")
    visualize_with_product(points, product,
                          title=f"Объект + чайник на ({center_x:.2f}, {center_y:.2f}, {max_z:.2f})")"""