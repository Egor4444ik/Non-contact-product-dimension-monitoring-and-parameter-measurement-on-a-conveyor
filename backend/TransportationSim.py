import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os

from ObjectEditing import MultyDimentionObject


class Simulation:
    def __init__(self, conveyor_point_step = 1, teapot_point_step = 1):
        teapot_filename = "teapot.txt"
        conveyor_filename = "ConveyorBeltAssemblyLine.txt"
        
        self.conveyor = MultyDimentionObject(conveyor_filename)
        self.conveyor.points = self.conveyor.points[::conveyor_point_step]
        self.conveyor.rotate_points(270, axis="x")
        
        self.scene_y_min = np.min(self.conveyor.points[:, 1])
        self.scene_y_max = np.max(self.conveyor.points[:, 1])
        
        self.conveyor_center_x = np.mean(self.conveyor.points[:, 0])
        self.conveyor_center_z = np.mean(self.conveyor.points[:, 2])
        
        teapot_target_size = np.ptp(self.conveyor.points, axis=0).max() / 10

        self.teapot = MultyDimentionObject(teapot_filename)
        self.teapot.points = self.teapot.points[::teapot_point_step]
        self.teapot.rotate_points(90, axis="x")
        self.teapot.object_scaling(teapot_target_size)

        teapot_min_z = np.min(self.teapot.points[:, 2])
        self.teapot_z_offset = np.max(self.conveyor.points[:, 2]) - teapot_min_z

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.scatter(self.conveyor.points[:, 0],
                        self.conveyor.points[:, 1],
                        self.conveyor.points[:, 2],
                        color='gray', alpha = 0.4 , s=0.5, zorder=1)

        self.teapot_scatter = self.ax.scatter(self.teapot.points[:, 0],
                                              self.teapot.points[:, 1],
                                              self.teapot.points[:, 2],
                                              color='red', s=0.5, zorder=2)

        self.ax.set_xlim(np.min(self.conveyor.points[:, 0]),
                         np.max(self.conveyor.points[:, 0]))
        self.ax.set_ylim(self.scene_y_min,
                         self.scene_y_max)
        self.ax.set_zlim(np.min(self.conveyor.points[:, 2]),
                         np.max(self.conveyor.points[:, 2]))
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

        ranges = np.ptp(self.conveyor.points, axis=0)
        aspect = ranges / ranges.max()
        self.ax.set_box_aspect(aspect)
        self.ax.view_init(azim=0)

        self.step = max(1, int((self.scene_y_max - self.scene_y_min) // 60))
        self.y_positions = list(range(int(self.scene_y_min) + 1,
                                      int(self.scene_y_max),
                                      self.step))

    def update(self, frame):
        y = self.y_positions[frame % len(self.y_positions)]
        target = np.array([self.conveyor_center_x, y, self.teapot_z_offset])
        self.teapot.replace(target)
        self.teapot_scatter._offsets3d =   (self.teapot.points[:, 0],
                                            self.teapot.points[:, 1],
                                            self.teapot.points[:, 2])
        return self.teapot_scatter,

    def run(self, all_scene_rerender = False):
        ani = FuncAnimation(self.fig, self.update,
                            frames=len(self.y_positions) * 2,
                            interval=50, blit=all_scene_rerender, repeat=True)
        plt.show()

    def save_frames(self, output_dir="frames", start_index=0):
        os.makedirs(output_dir, exist_ok=True)
        for i, y in enumerate(self.y_positions):
            target = np.array([self.conveyor_center_x, y, self.teapot_z_offset])
            self.teapot.replace(target)
            filename = os.path.join(output_dir, f"frame_{i + start_index:04d}.txt")
            all_points = np.vstack([self.teapot.points, self.conveyor.points])
            np.savetxt(filename, all_points, fmt='%.6f', delimiter=' ')
        print(f"Сохранено {len(self.y_positions)} кадров в папку '{output_dir}'")

if __name__ == '__main__':
    sim = Simulation(conveyor_point_step=20, teapot_point_step=10)
    sim.run(all_scene_rerender = False)
    #sim.save_frames("Teapots_production")