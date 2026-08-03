import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from scipy.spatial import KDTree

def are_coplanar(points, tol=1e-8):
    """Проверяет, лежат ли все точки в одной плоскости."""
    if points.shape[0] < 3:
        return True
    # Берём три первые точки для построения плоскости
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < tol:
        # первые три точки коллинеарны – ищем другие
        for i in range(3, points.shape[0]):
            normal = np.cross(points[i] - p0, p1 - p0)
            if np.linalg.norm(normal) > tol:
                break
        else:
            return True  # все точки на одной линии
    # Проверяем, что все точки лежат в этой плоскости
    for pt in points:
        if abs(np.dot(pt - p0, normal)) > tol:
            return False
    return True

def compute_volume(points, alpha_mult=28.0):
    print("alpha mult:", alpha_mult)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.remove_duplicated_points()
    tree = KDTree(points)
    dists, _ = tree.query(points, k=2)
    mean_dist = np.mean(dists[:, 1])
    alpha = mean_dist * alpha_mult
    print(f"alpha = {alpha:.4f}")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    if mesh.is_watertight():
        return mesh.get_volume(), mesh
    else:
        print(f"С {len(points)} точками и текущей плотностью точек работать нельзя")
        return None, None

def visualize_mesh(mesh, points=None):
    """Показывает сетку и, опционально, облако точек."""
    geometries = [mesh]
    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Окрасим точки в красный цвет для наглядности
        pcd.paint_uniform_color([1, 0, 0])
        geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries, window_name="Объект и сетка")

def read_points_from_file(filename, step):
    """
    Читает точки из текстового файла, где каждая строка содержит
    три координаты, разделённые пробелами или запятыми.
    """
    points = []
    with open(filename, 'r') as f:
        lines=f.readlines()
        for i in range(0, len(lines), step):
            line = lines[i].strip()
            parts = line.split()
            x, y, z = map(float, parts)
            points.append([x, y, z])
                
    return np.array(points)

if __name__ == "__main__":
    points = read_points_from_file("ConveyorBeltAssemblyLine.txt", 1)
    print("Точки получены")
    MIN_ALPHA = 21
    MAX_ALPHA = 30
    for step in range(1, 101):
        points_subset = points[::step]
        try:
            #for ALPHA_STEP in range(MIN_ALPHA*10, MAX_ALPHA*10):
            #    alpha_mult = ALPHA_STEP/10
            #    vol, mesh = compute_volume(points_subset, alpha_mult=alpha_mult)
            #    if step == 1 and vol:
            #        print(f"Наилучший объём при alpha mult {alpha_mult} равен: {vol}")
            vol, mesh = compute_volume(points_subset)
            visualize_mesh(mesh, points=points_subset)
            print(f"Объём ленточнфого конвейера при шаге {step} точек в общем количестве {len(points_subset)} точек: {vol}")
        except:
            print("Объём не рассчитался")
            print(f"Объём ленточнфого конвейера при шаге {step} точек в общем количестве {len(points_subset)} точек определить не возможно")
            break