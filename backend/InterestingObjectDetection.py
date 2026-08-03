import numpy as np
import pyvista

from pathlib import Path
import os

from backend.utils.config import OBJECTS_OF_INTEREST_FOLDER, PRODUCTION_DATA


class ObjectExtraction:
    def __init__(self):
        parent_teapots_folder = Path(PRODUCTION_DATA)
        filepaths = [file for file in os.listdir(parent_teapots_folder)]
        filepaths.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.input_frames = [np.loadtxt(parent_teapots_folder/Path(filepath)) for filepath in filepaths]
        lenghts = sum([len(input_frame) for input_frame in self.input_frames])
        print(f"Общее количество трёхмерных точек всей съёмки на камеру глубины: {lenghts}")
        self.frames = []
        self.diff_frames = []

        self.obj_y_center = np.mean(self.input_frames[0][:, 1])

    def frame_extacter(self):
        vicinityphi = 200
        left_bound, right_bound = self.obj_y_center - vicinityphi, self.obj_y_center + vicinityphi
        current_frames = self.input_frames

        for input_frame in self.input_frames:
            mask = (input_frame[:, 1] >= left_bound) & (input_frame[:, 1] <= right_bound)
            self.frames.append(input_frame[mask])

        low_extremums = []
        high_extremums = []
        prev_frame_len = len(self.frames[0])
        detection_started = False
        for i, frame in enumerate(self.frames):
            current_frame_len = len(frame)
            print(f"Длина {i} кадра: {current_frame_len}")
            
            if prev_frame_len < current_frame_len and not detection_started:
                detection_started = True
                start_frame_index = i - 1
            elif prev_frame_len > current_frame_len and detection_started:
                low_extremums.append(start_frame_index)
                high_extremums.append(i - 1)
                detection_started = False

            prev_frame_len = current_frame_len

        low_extremum_frames = [self.frames[i] for i in low_extremums]
        high_extremum_frames = [self.frames[i] for i in high_extremums]

        for min_frame, max_frame in zip(low_extremum_frames, high_extremum_frames):
            dtype = np.dtype([('x', float), ('y', float), ('z', float)])
            min_struct = np.array([tuple(row) for row in min_frame], dtype=dtype)
            max_struct = np.array([tuple(row) for row in max_frame], dtype=dtype)
            mask = np.isin(max_struct, min_struct)
            diff_mask = ~mask
            self.diff_frames.append(max_frame[diff_mask])

    def visualize_dif_obj(self):
        for diff_frame in self.diff_frames:
            plotter = pyvista.Plotter(window_size=(1000, 700))
            plotter.add_points(diff_frame, color='blue', point_size=3, label='Объект (точки)')
            plotter.add_legend()
            plotter.set_background('white')
            plotter.add_axes()
            plotter.show(interactive=True)

    def save_frames(self, output_dir=OBJECTS_OF_INTEREST_FOLDER, start_index=0):
        os.makedirs(output_dir, exist_ok=True)
        for i, diff_frame in enumerate(self.diff_frames):
            filename = os.path.join(output_dir, f"{OBJECTS_OF_INTEREST_FOLDER[:-1]}_{i + start_index:04d}.txt")
            np.savetxt(filename, diff_frame, fmt='%.6f', delimiter=' ')
        print(f"Сохранено {len(self.diff_frames)} кадров в папку '{output_dir}'")

if __name__ == '__main__':
    obj = ObjectExtraction()
    obj.frame_extacter()
    obj.save_frames()
    obj.visualize_dif_obj()


