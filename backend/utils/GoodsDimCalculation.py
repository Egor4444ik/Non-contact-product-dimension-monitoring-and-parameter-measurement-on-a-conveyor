import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from typing import Optional, Tuple, Union, List


class PointCloudVolumeAnalyzer:
    """
    Класс для анализа объёма замкнутой поверхности по облаку точек.
    """

    def __init__(self, points: Optional[np.ndarray] = None,
                 filepath: Optional[str] = None,
                 alpha_mult: float = 28.0):
        """
        Инициализация анализатора.

        :param points: массив точек (N x 3) – если передан, используется как основные точки.
        :param filepath: путь к текстовому файлу с точками (если points не указан).
        :param alpha_mult: множитель для автоматического подбора alpha (используется при вычислении объёма).
        """
        self.__points = np.empty((0, 3))
        self.__mesh = None
        self.__volume = None
        self.__alpha_mult = alpha_mult

        if points is not None:
            self.points = points
        elif filepath is not None:
            self.load_from_file(filepath)
        else:
            self.__points = np.empty((0, 3))

    # ---------------------- Свойства ----------------------

    @property
    def points(self) -> np.ndarray:
        """Текущий набор точек (копия)."""
        return self.__points.copy()

    @points.setter
    def points(self, value: np.ndarray) -> None:
        """Установить точки с проверкой размерности."""
        if not isinstance(value, np.ndarray):
            raise TypeError("Points must be a numpy array")
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        self.__points = value.astype(np.float64)
        self.__mesh = None
        self.__volume = None

    @property
    def volume(self) -> Optional[float]:
        """Вычисленный объём (если был вызван compute_volume)."""
        if self.__volume is None and self.__points.size > 0:
            self.compute_volume()
        return self.__volume

    @property
    def mesh(self):
        """Построенная полигональная сетка (если была вычислена)."""
        if self.__mesh is None and self.__points.size > 0:
            self.compute_volume()
        return self.__mesh

    @property
    def alpha_mult(self) -> float:
        """Множитель для автоматического определения alpha."""
        return self.__alpha_mult

    @alpha_mult.setter
    def alpha_mult(self, value: float) -> None:
        if value <= 0:
            raise ValueError("alpha_mult must be positive")
        self.__alpha_mult = float(value)
        self.__mesh = None
        self.__volume = None

    @property
    def point_count(self) -> int:
        """Количество точек."""
        return self.__points.shape[0]

    # ---------------------- Публичные методы ----------------------

    def load_from_file(self, filepath: str, step: int = 1) -> None:
        """
        Загрузить точки из текстового файла (каждая строка: x y z).
        :param filepath: путь к файлу.
        :param step: шаг прореживания (берутся каждая step-я строка).
        """
        points = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), step):
                line = lines[i].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                x, y, z = map(float, parts[:3])
                points.append((x, y, z))
        self.points = np.array(points, dtype=np.float64)

    def compute_volume(self, alpha_mult: Optional[float] = None) -> Optional[float]:
        """
        Вычислить объём замкнутой поверхности на основе текущих точек.
        Использует Poisson surface reconstruction.

        :param alpha_mult: (опционально) новый множитель alpha. Если задан, обновляет self.alpha_mult.
        :return: объём или None, если построить замкнутую сетку не удалось.
        """
        if self.point_count == 0:
            print("Нет точек для вычисления объёма.")
            return None

        if alpha_mult is not None:
            self.alpha_mult = alpha_mult

        # Создаём облако точек и удаляем дубликаты
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.__points)
        pcd.remove_duplicated_points()

        # Оценка среднего расстояния до ближайшего соседа
        tree = KDTree(self.__points)
        dists, _ = tree.query(self.__points, k=2)
        mean_dist = np.mean(dists[:, 1])
        alpha = mean_dist * self.__alpha_mult
        print(f"alpha_mult = {self.__alpha_mult:.2f}, alpha = {alpha:.6f}")

        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )
        except Exception as e:
            print(f"Ошибка при реконструкции: {e}")
            return None

        if mesh is None or not mesh.is_watertight():
            print(f"С {self.point_count} точками и текущей плотностью работать нельзя")
            self.__mesh = None
            self.__volume = None
            return None

        # Сохраняем результат
        self.__mesh = mesh
        self.__volume = mesh.get_volume()
        return self.__volume

    def visualize(self, show_points: bool = True) -> None:
        """
        Визуализировать построенную сетку (и, опционально, исходные точки).
        """
        if self.mesh is None:
            print("Сетка не построена. Сначала выполните compute_volume().")
            return

        geometries = [self.__mesh]
        if show_points and self.point_count > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.__points)
            pcd.paint_uniform_color([1, 0, 0])  # красные точки
            geometries.append(pcd)

        o3d.visualization.draw_geometries(geometries, window_name="Объект и сетка")

    def subsample(self, step: int) -> 'PointCloudVolumeAnalyzer':
        """
        Создать новый анализатор с прореженным набором точек.

        :param step: шаг прореживания (берутся точки с индексом, кратным step).
        :return: новый экземпляр класса с подвыборкой.
        """
        if step <= 0:
            raise ValueError("step must be positive")
        if self.point_count == 0:
            return PointCloudVolumeAnalyzer()
        subset = self.__points[::step]
        new_analyzer = PointCloudVolumeAnalyzer(subset)
        new_analyzer.alpha_mult = self.alpha_mult
        return new_analyzer

    @staticmethod
    def are_coplanar(points: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Проверяет, лежат ли все точки в одной плоскости.

        :param points: массив (N, 3).
        :param tol: допуск на отклонение от плоскости.
        :return: True, если все точки компланарны.
        """
        if points.shape[0] < 3:
            return True
        p0 = points[0]
        p1 = points[1]
        p2 = points[2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < tol:
            for i in range(3, points.shape[0]):
                normal = np.cross(points[i] - p0, p1 - p0)
                if np.linalg.norm(normal) > tol:
                    break
            else:
                return True
        for pt in points:
            if abs(np.dot(pt - p0, normal)) > tol:
                return False
        return True


# ---------------------- Точка входа ----------------------

if __name__ == "__main__":
    file_path = "ConveyorBeltAssemblyLine.txt"
    try:
        # Загружаем полный набор точек
        full_analyzer = PointCloudVolumeAnalyzer(filepath=file_path)
        print(f"Загружено {full_analyzer.point_count} точек.")

        # Диапазон шагов для исследования
        for step in range(1, 101):
            # Создаём анализатор с подвыборкой
            sub_analyzer = full_analyzer.subsample(step)
            print(f"\nШаг {step}: точек = {sub_analyzer.point_count}")

            # Вычисляем объём
            vol = sub_analyzer.compute_volume()
            if vol is not None:
                print(f"  Объём = {vol:.6f}")
                # Визуализация
                # sub_analyzer.visualize()
            else:
                print("  Объём не рассчитан (не удалось построить замкнутую сетку)")
                break

    except Exception as e:
        print(f"Ошибка: {e}")