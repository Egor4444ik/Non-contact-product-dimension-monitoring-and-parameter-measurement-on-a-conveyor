import math
import random
import sys
import argparse
from collections import defaultdict

def parse_obj(filename):
    """
    Парсит OBJ-файл. Возвращает:
        vertices: список кортежей (x, y, z) с плавающей запятой
        faces: список списков целых индексов вершин (0-индексация)
    """
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                # вершина: v x y z [w]
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif parts[0] == 'f':
                # грань: f v1[/vt1][/vn1] v2[/vt2][/vn2] ...
                # извлекаем только индексы вершин (до первого '/')
                face_indices = []
                for token in parts[1:]:
                    idx = token.split('/')[0]
                    if idx:
                        face_indices.append(int(idx) - 1)  # OBJ индексация с 1
                if len(face_indices) >= 3:
                    faces.append(face_indices)
    return vertices, faces

def triangulate_faces(faces):
    """
    Разбивает многоугольные грани на треугольники (веерная триангуляция).
    Возвращает список треугольников (каждый – список из 3 индексов).
    """
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) > 3:
            # веер от первой вершины
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i+1]])
        # игнорируем грани с < 3 вершинами
    return triangles

def triangle_area(v0, v1, v2):
    """Площадь треугольника по трём вершинам (векторное произведение)."""
    ax, ay, az = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    bx, by, bz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    return 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)

def generate_points_on_triangle(v0, v1, v2, n):
    """
    Генерирует n точек равномерно распределённых по площади треугольника.
    Использует метод барицентрических координат с коррекцией sqrt для равномерности.
    Возвращает список кортежей (x,y,z).
    """
    points = []
    for _ in range(n):
        r1 = random.random()
        r2 = random.random()
        # Преобразование для равномерного распределения по площади
        sqrt_r1 = math.sqrt(r1)
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2
        # Координаты точки
        x = a*v0[0] + b*v1[0] + c*v2[0]
        y = a*v0[1] + b*v1[1] + c*v2[1]
        z = a*v0[2] + b*v1[2] + c*v2[2]
        points.append((x, y, z))
    return points

def sample_surface(vertices, triangles, density=None, num_points=None):
    """
    Дискретизирует поверхность, состоящую из треугольников, облаком точек.
    Параметры:
        vertices: список вершин
        triangles: список треугольников (индексы вершин)
        density: число точек на единицу площади (если None, игнорируется)
        num_points: общее желаемое число точек (если None, вычисляется из density)
    Возвращает список точек (x,y,z).
    """
    if density is None and num_points is None:
        raise ValueError("Необходимо задать либо density, либо num_points")
    
    # Вычисляем площади всех треугольников
    areas = []
    total_area = 0.0
    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        area = triangle_area(v0, v1, v2)
        areas.append(area)
        total_area += area
    
    if total_area == 0:
        return []  # поверхность вырождена
    
    # Определяем общее количество точек
    if num_points is not None:
        N = num_points
    else:
        N = int(round(density * total_area))
        if N < 1:
            N = 1  # хотя бы одна точка
    
    # Распределяем точки пропорционально площадям
    all_points = []
    remaining = N
    for i, tri in enumerate(triangles):
        # Число точек для этого треугольника (пропорционально площади)
        if i == len(triangles) - 1:
            n_i = remaining  # отдаём остаток последнему
        else:
            n_i = int(round(N * areas[i] / total_area))
            if n_i < 0:
                n_i = 0
        # Генерация точек
        if n_i > 0:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            pts = generate_points_on_triangle(v0, v1, v2, n_i)
            all_points.extend(pts)
            remaining -= n_i
    
    # Если из-за округления остались точки, добавим их к последнему треугольнику (уже сделано)
    # Но для надёжности можно перераспределить, но обычно достаточно.
    return all_points

def save_points(points, filename):
    """Сохраняет точки в текстовый файл (по одной координате на строку)."""
    with open(filename, 'w') as f:
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

def main():
    density = 10
    num_points = None
    output = "teapot.txt"#"ConveyorBeltAssemblyLine.txt"

    filename = "teapot.obj"#"uploads_files_4403155_Belt+assembly+line copy.obj"
    try:
        vertices, faces = parse_obj(filename)
        if not vertices or not faces:
            print("Ошибка: файл не содержит вершин или граней", file=sys.stderr)
            sys.exit(1)
        
        triangles = triangulate_faces(faces)
        print(f"Загружено {len(vertices)} вершин, {len(faces)} граней, триангулировано в {len(triangles)} треугольников.")
        
        points = sample_surface(vertices, triangles, density=density, num_points=num_points)
        print(f"Сгенерировано {len(points)} точек.")
        
        save_points(points, output)
        print(f"Точки сохранены в {output}")
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()