import numpy as np

class OBJAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.vertices = []
        self.colors = []
        self.faces = []

    def parse_obj(self):
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('mtllib'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # Строки вершин
                if parts[0] == 'v':
                    # v x y z [r g b]
                    coords = list(map(float, parts[1:]))
                    self.vertices.append(coords)
                
                # Строки граней
                elif parts[0] == 'f':
                    
                    face_vertices = []
                    for vertex in parts[1:]:
                        # Форматирование строк вида v//vt//vn
                        indices = vertex.split('/')
                        v_idx = int(indices[0]) - 1  # Нормализация индексов на минус 1
                        face_vertices.append(v_idx)
                    
                    # Триангулируем полигон
                    for i in range(1, len(face_vertices) - 1):
                        self.faces.append([
                            face_vertices[0],
                            face_vertices[i],
                            face_vertices[i + 1]
                        ])
                                
    
    def get_point_density(self): # Определение плотности точек
        vertices_array = np.array(self.vertices)
        
        # Рассчёт ограничивающего объема
        min_coords = vertices_array.min(axis=0)
        max_coords = vertices_array.max(axis=0)
        
        # Рассчёт объема ограничивающих рамок
        volume = np.prod(max_coords - min_coords)

        # Рассчёт плотности точек
        density = len(self.vertices) / volume if volume != 0 else 0
        return density
    
    def get_point_count(self): # Определение количества точек
        return len(self.vertices)
    
    def get_face_count(self): # Определение количества граней
        return len(self.faces)
    
    def analyze(self):
        self.parse_obj()
        
        return {
            'point_count': self.get_point_count(),
            'face_count': self.get_face_count(),
            'point_density': self.get_point_density()
        }

if __name__ == "__main__":
    obj = "uploads_files_4403155_Belt+assembly+line copy.obj"

    analyzer = OBJAnalyzer(obj)
    result = analyzer.analyze()
    
    print("=" * 50)
    print("Результаты анализа OBJ файла:")
    print("=" * 50)
    print(f"Количество вершин (точек): {result['point_count']}")
    print(f"Плотность точек: {result['point_density']:.6f} точек/м³")
    print(f"Количество граней: {result['face_count']}")
    """

    print("=" * 50)
    print("Результаты анализа OBJ файла:")
    print("=" * 50)
    print(f"Количество вершин (точек): {result['point_count']}")
    print(f"Плотность точек: {result['point_density']:.6f} точек/ед³")
    print(f"Количество граней: {result['face_count']}")
    print(f"Наличие цветов вершин: {result['has_colors']}")
    print(f"Наличие материалов: {result['has_materials']}")
    print("\nИнформация о цветовой модели:")
    color_info = result['color_model']
    print(f"  Модель: {color_info['color_model']}")
    print(f"  Цветовое пространство: {color_info['color_space']}")
    print(f"  Детали: {color_info['details']}")"""