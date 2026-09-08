import math
import random
import sys
from collections import Counter

def parse_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif parts[0] == 'f':
                face_indices = []
                for token in parts[1:]:
                    idx = token.split('/')[0]
                    if idx:
                        face_indices.append(int(idx) - 1)
                if len(face_indices) >= 3:
                    faces.append(face_indices)
    return vertices, faces

def triangulate_faces(faces, vertices):
    def dist(i, j):
        v1, v2 = vertices[i], vertices[j]
        return math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
    
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            v0, v1, v2, v3 = face
            d02 = dist(v0, v2)
            d13 = dist(v1, v3)
            if d02 < d13:
                triangles.append([v0, v1, v2])
                triangles.append([v0, v2, v3])
            else:
                triangles.append([v1, v2, v3])
                triangles.append([v1, v3, v0])
        else:
            for i in range(1, len(face)-1):
                triangles.append([face[0], face[i], face[i+1]])
    return triangles

def triangle_area(v0, v1, v2):
    ax, ay, az = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    bx, by, bz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    return 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)

def generate_points_on_triangle(v0, v1, v2, n):
    points = []
    for _ in range(n):
        r1 = random.random()
        r2 = random.random()
        sqrt_r1 = math.sqrt(r1)
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2
        x = a*v0[0] + b*v1[0] + c*v2[0]
        y = a*v0[1] + b*v1[1] + c*v2[1]
        z = a*v0[2] + b*v1[2] + c*v2[2]
        points.append((x, y, z))
    return points

def sample_surface(vertices, triangles, density=None, num_points=None):
    if density is None and num_points is None:
        raise ValueError("Необходимо задать либо density, либо num_points")
    
    # Вычисляем площади и отбрасываем вырожденные
    valid_tris = []
    valid_areas = []
    total_area = 0.0
    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        area = triangle_area(v0, v1, v2)
        if area > 1e-12:   # фильтр нулевых площадей
            valid_tris.append(tri)
            valid_areas.append(area)
            total_area += area
    
    print(f"Всего треугольников: {len(triangles)}, из них с ненулевой площадью: {len(valid_tris)}")
    print(f"Суммарная площадь: {total_area:.6f}")
    
    if total_area == 0:
        return []
    
    if num_points is not None:
        N = num_points
    else:
        N = int(round(density * total_area))
        if N < 1:
            N = 1
    
    all_points = []
    remaining = N
    for i, tri in enumerate(valid_tris):
        if i == len(valid_tris) - 1:
            n_i = remaining
        else:
            n_i = int(round(N * valid_areas[i] / total_area))
            if n_i < 0:
                n_i = 0
        if n_i > 0:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            pts = generate_points_on_triangle(v0, v1, v2, n_i)
            all_points.extend(pts)
            remaining -= n_i
    
    return all_points

def save_points(points, filename):
    with open(filename, 'w') as f:
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

def main():
    density = None
    num_points = 3000000
    output = "barbell.txt"
    filename = "barbell.OBJ"
    
    try:
        vertices, faces = parse_obj(filename)
        if not vertices or not faces:
            print("Ошибка: файл не содержит вершин или граней", file=sys.stderr)
            sys.exit(1)
        
        triangles = triangulate_faces(faces, vertices)
        print(f"Загружено {len(vertices)} вершин, {len(faces)} граней, триангулировано в {len(triangles)} треугольников.")
        
        points = sample_surface(vertices, triangles, density=density, num_points=num_points)
        points = points[::120]
        print(f"Сгенерировано {len(points)} точек.")
        save_points(points, output)
        print(f"Точки сохранены в {output}")
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()