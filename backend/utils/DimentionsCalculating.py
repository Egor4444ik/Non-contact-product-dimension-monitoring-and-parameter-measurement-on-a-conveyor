import numpy as np
from typing import List, Tuple, Optional, Any, Dict

class OBJAnalyzer:
    """
    Анализатор OBJ-файлов
    """
    def __init__(self, filepath: str):
        self.__filepath = filepath
        self.__vertices: List[List[float]] = []     
        self.__colors: List[Optional[List[float]]] = []
        self.__faces: List[List[int]] = []            
        self.__has_materials: bool = False            
        self.__parsed: bool = False                   

    # ---------------------- Свойства (геттеры/сеттеры) ----------------------

    @property
    def filepath(self) -> str:
        """Путь к файлу (только чтение)."""
        return self.__filepath

    @property
    def vertices(self) -> List[List[float]]:
        """Список координат вершин (только чтение, возвращается копия)."""
        self.__ensure_parsed()
        return [v[:] for v in self.__vertices]

    @vertices.setter
    def vertices(self, new_vertices: List[List[float]]) -> None:
        """
        Установить новый список вершин с проверкой размерности.
        Каждая вершина должна содержать 3 числа (x, y, z).
        """
        if not isinstance(new_vertices, list):
            raise TypeError("vertices must be a list")
        for v in new_vertices:
            if not (isinstance(v, (list, tuple)) and len(v) == 3):
                raise ValueError("Each vertex must contain exactly 3 coordinates")
        self.__vertices = [list(v) for v in new_vertices]
        self.__parsed = True  # если устанавливаем вручную, считаем, что данные загружены

    @property
    def colors(self) -> List[Optional[List[float]]]:
        """Список цветов вершин (копия)."""
        self.__ensure_parsed()
        return [c[:] if c is not None else None for c in self.__colors]

    @colors.setter
    def colors(self, new_colors: List[Optional[List[float]]]) -> None:
        """Установить цвета с проверкой размерности (3 числа)."""
        if not isinstance(new_colors, list):
            raise TypeError("colors must be a list")
        for c in new_colors:
            if c is not None:
                if not (isinstance(c, (list, tuple)) and len(c) == 3):
                    raise ValueError("Each color must contain exactly 3 components")
        self.__colors = [list(c) if c is not None else None for c in new_colors]

    @property
    def faces(self) -> List[List[int]]:
        """Список граней (индексы вершин) - копия."""
        self.__ensure_parsed()
        return [f[:] for f in self.__faces]

    @faces.setter
    def faces(self, new_faces: List[List[int]]) -> None:
        """Установить грани с проверкой на целые числа."""
        if not isinstance(new_faces, list):
            raise TypeError("faces must be a list")
        for f in new_faces:
            if not all(isinstance(i, int) for i in f):
                raise ValueError("Face indices must be integers")
        self.__faces = [f[:] for f in new_faces]

    @property
    def point_count(self) -> int:
        """Количество вершин."""
        self.__ensure_parsed()
        return len(self.__vertices)

    @property
    def face_count(self) -> int:
        """Количество граней (треугольников)."""
        self.__ensure_parsed()
        return len(self.__faces)

    @property
    def point_density(self) -> float:
        """Плотность точек (количество вершин на единицу объёма)."""
        self.__ensure_parsed()
        if not self.__vertices:
            return 0.0
        arr = np.array(self.__vertices)
        min_coords = arr.min(axis=0)
        max_coords = arr.max(axis=0)
        volume = np.prod(max_coords - min_coords)
        return len(self.__vertices) / volume if volume != 0 else 0.0

    @property
    def has_colors(self) -> bool:
        """Есть ли у вершин цвета."""
        self.__ensure_parsed()
        return any(c is not None for c in self.__colors)

    @property
    def has_materials(self) -> bool:
        """Присутствует ли ссылка на файл материалов (mtllib)."""
        self.__ensure_parsed()
        return self.__has_materials

    @property
    def color_model(self) -> Dict[str, str]:
        """Информация о цветовой модели (заглушка для совместимости с комментариями)."""
        self.__ensure_parsed()
        if self.has_colors:
            return {
                'color_model': 'RGB',
                'color_space': 'sRGB',
                'details': 'Цвета заданы для каждой вершины'
            }
        else:
            return {
                'color_model': 'None',
                'color_space': 'N/A',
                'details': 'Цвета не указаны'
            }

    # ---------------------- Внутренние методы ----------------------

    def __ensure_parsed(self) -> None:
        """Выполнить парсинг файла, если он ещё не был сделан."""
        if not self.__parsed:
            self.__parse_obj()

    def __parse_obj(self) -> None:
        """
        Парсинг OBJ-файла: заполняет приватные списки вершин, цветов и граней.
        Вызывается автоматически при первом обращении к данным.
        """
        with open(self.__filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                if parts[0] == 'mtllib':
                    self.__has_materials = True
                    continue

                if parts[0] == 'v':
                    
                    coords = list(map(float, parts[1:]))
                    if len(coords) == 3:
                        self.__vertices.append(coords)
                        self.__colors.append(None)
                    elif len(coords) >= 6:
                        self.__vertices.append(coords[:3])
                        self.__colors.append(coords[3:6])
                    else:
                        continue

                elif parts[0] == 'f':
                    # Парсинг граней
                    face_vertices = []
                    for token in parts[1:]:
                        indices = token.split('/')
                        try:
                            v_idx = int(indices[0]) - 1
                            face_vertices.append(v_idx)
                        except ValueError:
                            continue
                    # Триангуляция полигона
                    for i in range(1, len(face_vertices) - 1):
                        self.__faces.append([
                            face_vertices[0],
                            face_vertices[i],
                            face_vertices[i + 1]
                        ])

        self.__parsed = True

    # ---------------------- Публичные методы ----------------------

    def add_vertex(self, x: float, y: float, z: float, color: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Добавить одну вершину (и опционально цвет) в конец списка.
        """
        self.__ensure_parsed()
        self.__vertices.append([x, y, z])
        self.__colors.append(list(color) if color else None)

    def add_face(self, v1: int, v2: int, v3: int) -> None:
        """
        Добавить треугольную грань по индексам вершин.
        """
        self.__ensure_parsed()
        self.__faces.append([v1, v2, v3])

    def analyze(self) -> Dict[str, Any]:
        """
        Выполнить полный анализ и вернуть словарь с результатами.
        """
        self.__ensure_parsed()
        return {
            'point_count': self.point_count,
            'face_count': self.face_count,
            'point_density': self.point_density,
            'has_colors': self.has_colors,
            'has_materials': self.has_materials,
            'color_model': self.color_model
        }

    def __repr__(self) -> str:
        return (f"OBJAnalyzer(filepath='{self.__filepath}', "
                f"vertices={self.point_count}, faces={self.face_count})")


# ---------------------- Точка входа ----------------------

if __name__ == "__main__":
    obj_file = "uploads_files_4403155_Belt+assembly+line copy.obj"
    analyzer = OBJAnalyzer(obj_file)
    result = analyzer.analyze()

    print("=" * 50)
    print("Результаты анализа OBJ файла:")
    print("=" * 50)
    print(f"Количество вершин (точек): {result['point_count']}")
    print(f"Плотность точек: {result['point_density']:.6f} точек/м³")
    print(f"Количество граней: {result['face_count']}")
    print(f"Наличие цветов вершин: {result['has_colors']}")
    print(f"Наличие материалов: {result['has_materials']}")
    print("\nИнформация о цветовой модели:")
    color_info = result['color_model']
    print(f"  Модель: {color_info['color_model']}")
    print(f"  Цветовое пространство: {color_info['color_space']}")
    print(f"  Детали: {color_info['details']}")