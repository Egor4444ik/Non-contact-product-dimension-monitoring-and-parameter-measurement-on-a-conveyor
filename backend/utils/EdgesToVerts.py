import math
import random
import sys
from typing import List, Tuple, Optional, Union, Dict, Any

class OBJSurfaceSampler:
    """
    Класс для генерации случайных точек на поверхности 3D-модели из OBJ-файла.
    """
    
    def __init__(self, filepath: str):
        self.__filepath = filepath
        self.__vertices: List[Tuple[float, float, float]] = []   # исходные вершины
        self.__faces: List[List[int]] = []                       # грани (полигоны) с индексами
        self.__triangles: List[List[int]] = []                   # триангулированные грани (треугольники)
        self.__parsed: bool = False
        self.__triangulated: bool = False
    
    # ---------------------- Свойства (геттеры/сеттеры) ----------------------
    
    @property
    def filepath(self) -> str:
        """Путь к файлу (только чтение)."""
        return self.__filepath
    
    @property
    def vertices(self) -> List[Tuple[float, float, float]]:
        """Список вершин (копия)."""
        self.__ensure_parsed()
        return self.__vertices[:]
    
    @vertices.setter
    def vertices(self, new_vertices: List[Tuple[float, float, float]]) -> None:
        """Установить список вершин с проверкой формата."""
        if not isinstance(new_vertices, list):
            raise TypeError("vertices must be a list")
        for v in new_vertices:
            if not (isinstance(v, (list, tuple)) and len(v) == 3):
                raise ValueError("Each vertex must have exactly 3 coordinates")
        self.__vertices = [tuple(v) for v in new_vertices]
        self.__parsed = True   # если установили вручную, считаем данные загруженными
    
    @property
    def faces(self) -> List[List[int]]:
        """Список граней (копия)."""
        self.__ensure_parsed()
        return [f[:] for f in self.__faces]
    
    @faces.setter
    def faces(self, new_faces: List[List[int]]) -> None:
        """Установить грани с проверкой индексов."""
        if not isinstance(new_faces, list):
            raise TypeError("faces must be a list")
        for f in new_faces:
            if not all(isinstance(i, int) for i in f):
                raise ValueError("Face indices must be integers")
        self.__faces = [f[:] for f in new_faces]
    
    @property
    def triangles(self) -> List[List[int]]:
        """Триангулированные грани (треугольники) – только чтение."""
        self.__ensure_triangulated()
        return [t[:] for t in self.__triangles]
    
    @property
    def vertex_count(self) -> int:
        """Количество вершин."""
        self.__ensure_parsed()
        return len(self.__vertices)
    
    @property
    def face_count(self) -> int:
        """Количество исходных граней."""
        self.__ensure_parsed()
        return len(self.__faces)
    
    @property
    def triangle_count(self) -> int:
        """Количество треугольников после триангуляции."""
        self.__ensure_triangulated()
        return len(self.__triangles)
    
    # ---------------------- Внутренние методы ----------------------
    
    def __ensure_parsed(self) -> None:
        """Выполнить парсинг OBJ-файла, если ещё не сделан."""
        if not self.__parsed:
            self.__parse_obj()
    
    def __ensure_triangulated(self) -> None:
        """Выполнить триангуляцию, если ещё не сделана."""
        self.__ensure_parsed()
        if not self.__triangulated:
            self.__triangulate_faces()
    
    def __parse_obj(self) -> None:
        """
        Парсинг OBJ-файла: заполняет __vertices и __faces.
        """
        with open(self.__filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    try:
                        x, y, z = map(float, parts[1:4])
                        self.__vertices.append((x, y, z))
                    except ValueError:
                        continue  # пропускаем некорректные строки
                
                elif parts[0] == 'f':
                    face_indices = []
                    for token in parts[1:]:
                        idx = token.split('/')[0]
                        if idx:
                            try:
                                face_indices.append(int(idx) - 1)
                            except ValueError:
                                pass
                    if len(face_indices) >= 3:
                        self.__faces.append(face_indices)
        
        self.__parsed = True
    
    def __triangulate_faces(self) -> None:
        """
        Триангуляция всех граней (поддерживаются многоугольники).
        Для четырёхугольников выбирается более короткая диагональ.
        """
        self.__ensure_parsed()
        
        def dist(i: int, j: int) -> float:
            v1, v2 = self.__vertices[i], self.__vertices[j]
            return math.sqrt(
                (v1[0] - v2[0]) ** 2 +
                (v1[1] - v2[1]) ** 2 +
                (v1[2] - v2[2]) ** 2
            )
        
        triangles = []
        for face in self.__faces:
            if len(face) == 3:
                triangles.append(face[:])
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
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])
        
        self.__triangles = triangles
        self.__triangulated = True
    
    # ---------------------- Публичные методы ----------------------
    
    def compute_triangle_area(self, v0: Tuple[float, float, float],
                               v1: Tuple[float, float, float],
                               v2: Tuple[float, float, float]) -> float:
        """
        Вычисляет площадь треугольника по координатам трёх вершин.
        """
        ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
        bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    
    def generate_points_on_triangle(self, v0: Tuple[float, float, float],
                                    v1: Tuple[float, float, float],
                                    v2: Tuple[float, float, float],
                                    n: int) -> List[Tuple[float, float, float]]:
        """
        Генерирует n случайных точек, равномерно распределённых по площади треугольника.
        Используется метод барицентрических координат с коррекцией.
        """
        points = []
        for _ in range(n):
            r1 = random.random()
            r2 = random.random()
            sqrt_r1 = math.sqrt(r1)
            a = 1.0 - sqrt_r1
            b = sqrt_r1 * (1.0 - r2)
            c = sqrt_r1 * r2
            x = a * v0[0] + b * v1[0] + c * v2[0]
            y = a * v0[1] + b * v1[1] + c * v2[1]
            z = a * v0[2] + b * v1[2] + c * v2[2]
            points.append((x, y, z))
        return points
    
    def sample_surface(self, *, density: Optional[float] = None,
                       num_points: Optional[int] = None,
                       subsample_step: int = 1) -> List[Tuple[float, float, float]]:
        """
        Генерирует точки на поверхности модели.
        
        Параметры:
            density: плотность точек на единицу площади (если задана, вычисляет N = density * total_area)
            num_points: фиксированное количество точек (если задано, переопределяет density)
            subsample_step: шаг прореживания (например, 120 – оставляет каждую 120-ю точку)
        
        Возвращает список сгенерированных точек.
        """
        if density is None and num_points is None:
            raise ValueError("Необходимо задать либо density, либо num_points")
        
        self.__ensure_triangulated()
        vertices = self.__vertices
        triangles = self.__triangles
        
        # Вычисляем площади всех треугольников, отбрасывая вырожденные
        valid_tris = []
        valid_areas = []
        total_area = 0.0
        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            area = self.compute_triangle_area(v0, v1, v2)
            if area > 1e-12:
                valid_tris.append(tri)
                valid_areas.append(area)
                total_area += area
        
        print(f"Всего треугольников: {len(triangles)}, из них с ненулевой площадью: {len(valid_tris)}")
        print(f"Суммарная площадь: {total_area:.6f}")
        
        if total_area == 0:
            return []
        
        # Определяем общее количество точек
        if num_points is not None:
            N = num_points
        else:
            N = int(round(density * total_area))
            if N < 1:
                N = 1
        
        all_points = []
        remaining = N
        # Распределяем точки пропорционально площадям треугольников
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
                pts = self.generate_points_on_triangle(v0, v1, v2, n_i)
                all_points.extend(pts)
                remaining -= n_i
        
        # Прореживание, если задан шаг
        if subsample_step > 1:
            all_points = all_points[::subsample_step]
        
        return all_points
    
    def save_points(self, points: List[Tuple[float, float, float]], filename: str) -> None:
        """
        Сохраняет список точек в текстовый файл (каждая строка: x y z).
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for x, y, z in points:
                f.write(f"{x} {y} {z}\n")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Возвращает сводную информацию о модели.
        """
        self.__ensure_triangulated()
        return {
            'vertex_count': self.vertex_count,
            'face_count': self.face_count,
            'triangle_count': self.triangle_count,
            'filepath': self.filepath
        }
    
    def __repr__(self) -> str:
        return (f"OBJSurfaceSampler(filepath='{self.filepath}', "
                f"vertices={self.vertex_count}, triangles={self.triangle_count})")


# ---------------------- Точка входа (демонстрация) ----------------------

def main() -> None:
    # Параметры
    input_file = "barbell.OBJ"
    output_file = "barbell.txt"
    target_points = 3000000
    subsample = 120  # оставляем каждую 120-ю точку
    
    try:
        sampler = OBJSurfaceSampler(input_file)
        
        # Можно посмотреть информацию
        info = sampler.analyze()
        print(f"Загружено {info['vertex_count']} вершин, {info['face_count']} граней, "
              f"триангулировано в {info['triangle_count']} треугольников.")
        
        # Генерируем точки
        points = sampler.sample_surface(num_points=target_points, subsample_step=subsample)
        print(f"Сгенерировано {len(points)} точек.")
        
        sampler.save_points(points, output_file)
        print(f"Точки сохранены в {output_file}")
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()