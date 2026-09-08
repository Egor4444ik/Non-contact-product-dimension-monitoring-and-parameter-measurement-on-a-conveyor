import pyvista
import numpy as np
import os
from pathlib import Path
from typing import Optional, Union

from .config import PRODUCTION_DATA


class PointCloudVisualizer:
    """
    Класс для загрузки и визуализации облака точек из текстовых файлов.
    """
    def __init__(self, filepath: Optional[Union[str, Path]] = None):
        """
        Инициализация визуализатора.

        :param filepath: путь к файлу с точками (опционально).
                         Если передан, точки загружаются сразу.
        """
        self.__points = np.empty((0, 3))   # приватный массив точек
        self.__filepath = None
        if filepath is not None:
            self.load_from_file(filepath)

    # ---------------------- Свойства (геттеры/сеттеры) ----------------------

    @property
    def points(self) -> np.ndarray:
        """Возвращает копию массива точек."""
        return self.__points.copy()

    @points.setter
    def points(self, value: np.ndarray) -> None:
        """
        Устанавливает новый массив точек с проверкой размерности.
        Должен быть массив формы (N, 3).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Точки должны быть numpy.ndarray")
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError("Массив точек должен иметь форму (N, 3)")
        self.__points = value.astype(np.float64)

    @property
    def filepath(self) -> Optional[Path]:
        """Путь к загруженному файлу (только чтение)."""
        return self.__filepath

    @property
    def point_count(self) -> int:
        """Количество точек."""
        return self.__points.shape[0]

    # ---------------------- Публичные методы ----------------------

    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """
        Загружает точки из текстового файла (каждая строка: x y z).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        self.__filepath = path
        self.__points = np.loadtxt(path)

    def visualize(self, color: str = 'blue', point_size: int = 3,
                  window_size: tuple = (1000, 700),
                  background: str = 'white',
                  show_axes: bool = True,
                  label: str = 'Объект (точки)',
                  interactive: bool = True) -> None:
        """
        Визуализирует текущее облако точек с помощью PyVista.

        :param color: цвет точек.
        :param point_size: размер точек.
        :param window_size: размер окна.
        :param background: цвет фона.
        :param show_axes: отображать оси координат.
        :param label: подпись в легенде.
        :param interactive: интерактивный режим.
        """
        if self.point_count == 0:
            print("Нет точек для визуализации.")
            return

        plotter = pyvista.Plotter(window_size=window_size)
        plotter.add_points(self.__points, color=color, point_size=point_size,
                           label=label)
        plotter.add_legend()
        plotter.set_background(background)
        if show_axes:
            plotter.add_axes()
        plotter.show(interactive=interactive)

    def __repr__(self) -> str:
        return (f"PointCloudVisualizer(filepath={self.filepath!r}, "
                f"point_count={self.point_count})")


# ---------------------- Точка входа (демонстрация) ----------------------

if __name__ == '__main__':
    files = [f for f in os.listdir(PRODUCTION_DATA) if os.path.isfile(PRODUCTION_DATA / f)]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for filename in files:
        filepath = PRODUCTION_DATA / filename
        print(filename)
        visualizer = PointCloudVisualizer(filepath)
        visualizer.visualize()