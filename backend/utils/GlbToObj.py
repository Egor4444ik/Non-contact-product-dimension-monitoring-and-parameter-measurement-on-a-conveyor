import sys
from typing import Optional, Tuple, List, Any
import trimesh
import numpy as np

class GLBToObjConverter:
    """
    Конвертер GLB-файлов в текстовый OBJ-формат.
    Поддерживает вершины, нормали, текстурные координаты и грани.
    Все данные хранятся в инкапсулированном виде.
    """
    
    def __init__(self, input_path: str):
        self.__input_path = input_path
        self.__mesh = None
        self.__vertices = np.array([])
        self.__faces = np.array([])
        self.__normals = np.array([])
        self.__texcoords = np.array([])
        self.__loaded = False
    
    # ---------------------- Приватные методы ----------------------
    
    def __ensure_loaded(self) -> None:
        """Загружает GLB-файл, если он ещё не загружен."""
        if not self.__loaded:
            self.__load()
    
    def __load(self) -> None:
        """
        Загружает геометрию из GLB-файла с помощью trimesh.
        Все компоненты объединяются в один объект.
        """
        mesh = trimesh.load(self.__input_path, force='mesh')
        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError("Геометрия не найдена или файл пуст.")
        
        self.__mesh = mesh
        self.__vertices = mesh.vertices.copy()
        self.__faces = mesh.faces.copy()
        
        # Нормали
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            self.__normals = mesh.vertex_normals.copy()
        else:
            self.__normals = np.array([])
        
        # Текстурные координаты
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            self.__texcoords = mesh.visual.uv.copy()
        else:
            self.__texcoords = np.array([])
        
        self.__loaded = True
    
    # ---------------------- Свойства (геттеры/сеттеры) ----------------------
    
    @property
    def vertices(self) -> np.ndarray:
        """Массив вершин (N x 3). Доступ только для чтения (возвращается копия)."""
        self.__ensure_loaded()
        return self.__vertices.copy()
    
    @vertices.setter
    def vertices(self, new_vertices: np.ndarray) -> None:
        """
        Установка нового массива вершин с проверкой размерности.
        Должен быть массив формы (N, 3).
        """
        if not isinstance(new_vertices, np.ndarray):
            raise TypeError("vertices must be a numpy array")
        if new_vertices.ndim != 2 or new_vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        self.__vertices = new_vertices.copy()
        self.__loaded = True  # считаем, что данные уже загружены (при ручной установке)
    
    @property
    def faces(self) -> np.ndarray:
        """Массив граней (M x 3). Только чтение."""
        self.__ensure_loaded()
        return self.__faces.copy()
    
    @faces.setter
    def faces(self, new_faces: np.ndarray) -> None:
        """Установка граней с проверкой формы (M, 3)."""
        if not isinstance(new_faces, np.ndarray):
            raise TypeError("faces must be a numpy array")
        if new_faces.ndim != 2 or new_faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        self.__faces = new_faces.copy()
    
    @property
    def normals(self) -> np.ndarray:
        """Массив нормалей (N x 3) или пустой массив, если нормали отсутствуют."""
        self.__ensure_loaded()
        return self.__normals.copy()
    
    @normals.setter
    def normals(self, new_normals: np.ndarray) -> None:
        """Установка нормалей с проверкой формы."""
        if new_normals.size == 0:
            self.__normals = np.array([])
            return
        if not isinstance(new_normals, np.ndarray):
            raise TypeError("normals must be a numpy array")
        if new_normals.ndim != 2 or new_normals.shape[1] != 3:
            raise ValueError("normals must have shape (N, 3)")
        self.__normals = new_normals.copy()
    
    @property
    def texcoords(self) -> np.ndarray:
        """Массив текстурных координат (N x 2) или пустой массив."""
        self.__ensure_loaded()
        return self.__texcoords.copy()
    
    @texcoords.setter
    def texcoords(self, new_texcoords: np.ndarray) -> None:
        """Установка текстурных координат с проверкой формы."""
        if new_texcoords.size == 0:
            self.__texcoords = np.array([])
            return
        if not isinstance(new_texcoords, np.ndarray):
            raise TypeError("texcoords must be a numpy array")
        if new_texcoords.ndim != 2 or new_texcoords.shape[1] != 2:
            raise ValueError("texcoords must have shape (N, 2)")
        self.__texcoords = new_texcoords.copy()
    
    @property
    def has_normals(self) -> bool:
        """Наличие нормалей."""
        self.__ensure_loaded()
        return self.__normals.size > 0
    
    @property
    def has_texcoords(self) -> bool:
        """Наличие текстурных координат."""
        self.__ensure_loaded()
        return self.__texcoords.size > 0
    
    @property
    def vertex_count(self) -> int:
        """Количество вершин."""
        self.__ensure_loaded()
        return self.__vertices.shape[0]
    
    @property
    def face_count(self) -> int:
        """Количество граней (треугольников)."""
        self.__ensure_loaded()
        return self.__faces.shape[0]
    
    # ---------------------- Публичные методы ----------------------
    
    def export_obj(self, output_path: str) -> None:
        """
        Экспортирует загруженную геометрию в текстовый файл в формате OBJ.
        Формат строк: v, vn, vt, f (с учётом наличия данных).
        """
        self.__ensure_loaded()
        
        vertices = self.__vertices
        faces = self.__faces
        normals = self.__normals
        texcoords = self.__texcoords
        
        has_vn = self.has_normals
        has_vt = self.has_texcoords
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Вершины
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Нормали
            if has_vn:
                for n in normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            # Текстурные координаты
            if has_vt:
                for t in texcoords:
                    f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
            
            # Грани (треугольники)
            for face in faces:
                # индексы в OBJ начинаются с 1
                i1, i2, i3 = face + 1
                if has_vn and has_vt:
                    f.write(f"f {i1}/{i1}/{i1} {i2}/{i2}/{i2} {i3}/{i3}/{i3}\n")
                elif has_vn and not has_vt:
                    f.write(f"f {i1}//{i1} {i2}//{i2} {i3}//{i3}\n")
                elif not has_vn and has_vt:
                    f.write(f"f {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")
                else:
                    f.write(f"f {i1} {i2} {i3}\n")
        
        print(f"Сохранено: {self.vertex_count} вершин, {self.face_count} граней → {output_path}")
    
    def get_info(self) -> dict:
        """
        Возвращает словарь с информацией о загруженной модели.
        """
        self.__ensure_loaded()
        return {
            'input_path': self.__input_path,
            'vertex_count': self.vertex_count,
            'face_count': self.face_count,
            'has_normals': self.has_normals,
            'has_texcoords': self.has_texcoords
        }
    
    def __repr__(self) -> str:
        return (f"GLBToObjConverter(input_path='{self.__input_path}', "
                f"vertices={self.vertex_count}, faces={self.face_count})")


# ---------------------- Точка входа (демонстрация) ----------------------

def main() -> None:
    input_glb = "teapot.glb"
    output_obj = "teapot.obj"
    
    try:
        converter = GLBToObjConverter(input_glb)
        info = converter.get_info()
        print(f"Загружено: {info['vertex_count']} вершин, {info['face_count']} граней")
        print(f"Нормали: {info['has_normals']}, Текстурные координаты: {info['has_texcoords']}")
        
        converter.export_obj(output_obj)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()