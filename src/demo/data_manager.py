from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils import CameraPose


class DataManager():
    """Manages trajectory and object data."""

    def __init__(self):

        self.trajectory = None
        self.objects_df = None
        self.floor_plane = None

        self.data_loaded = False

    def load_data(self,data_dir:Path):
        data_dir = Path(data_dir)
        self.trajectory = self.load_trajectory(data_dir / "incremental_analysis_detailed_trajectory.txt")
        
        df = pd.read_csv(data_dir / "incremental_analysis_detailed_closest_objects.csv")
        df = df.loc[df.groupby(['segment_id', 'step'])['closest_2d_distance'].idxmin()]
        self.objects_df = df.sort_values(by='step').reset_index(drop=True)

        #filter out background
        self.objects_df = self.objects_df.loc[self.objects_df.class_label != "background"]

        #clean classlabels
        self.objects_df.class_label = self.objects_df.class_label.apply(lambda x: x if ":" not in x else x.split(":",1)[0])

        # Load floor plane data
        floor_df = pd.read_csv(data_dir / "incremental_analysis_detailed_floor_data.csv")
        if not floor_df.empty:
            # Take the first detected floor
            row = floor_df.iloc[0]
            normal = np.array([row.normal_x, row.normal_y, row.normal_z])
            offset = -1 * row.offset #account for different definition of offset
            self.floor_plane = (normal, offset)

        self.data_loaded = True

    def load_trajectory(self, trajectory_file: Path) -> Dict[int, CameraPose]:
        """Load camera trajectory data."""
        trajectory = {}
        data = np.loadtxt(trajectory_file)
        
        for i, row in enumerate(data):
            position = row[1:4]
            quaternion = row[4:8]
            trajectory[i] = CameraPose.from_position_and_quaternion(position, quaternion)
        
        return trajectory
        
    def get_frame_data(self, step_idx: int):
        """Get camera pose and objects for a specific frame."""
        
        camera_pose = self.trajectory.get(step_idx)
        frame_objects = self.objects_df[self.objects_df.step == step_idx]
        return camera_pose, frame_objects

    def plot_step_static(self, step_idx: int, axis_length: float = 0.5):
        """Generate and return a filepath to a static 3D plot of the given step."""
        import matplotlib.pyplot as plt
        from pathlib import Path
        import tempfile
        import uuid

        pose, objects = self.get_frame_data(step_idx)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw camera trajectory
        positions = np.array([p.position for p in self.trajectory.values()])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='black', label='Trajectory')

        # Draw floor
        if self.floor_plane:
            normal, offset = self.floor_plane
            normal = normal / np.linalg.norm(normal)

            mid = positions.mean(axis=0)
            max_range = (positions.max(axis=0) - positions.min(axis=0)).max() / 2
            plane_size = max_range * 2
            xx, yy = np.meshgrid(
                np.linspace(mid[0] - plane_size, mid[0] + plane_size, 20),
                np.linspace(mid[1] - plane_size, mid[1] + plane_size, 20)
            )
            zz = (-normal[0] * xx - normal[1] * yy - offset) / normal[2]
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')

            # Set view to top-down
            def view_angles_from_normal(n):
                n = -n / np.linalg.norm(n)
                elev = np.arcsin(n[2])
                azim = np.arctan2(n[1], n[0])
                return np.degrees(elev), np.degrees(azim)

            elev, azim = view_angles_from_normal(normal)
            ax.view_init(elev=elev, azim=azim)

        if pose:
            pos = pose.position
            ax.scatter(pos[0], pos[1], pos[2], color='blue', s=60)

            rot = pose.rotation.as_matrix()
            for i, color in enumerate(['r', 'g', 'b']):
                end = pos + rot[:, i] * axis_length
                ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]], color=color, linewidth=2)

        if not objects.empty:
            xs = objects.closest_3d_x.values
            ys = objects.closest_3d_y.values
            zs = objects.closest_3d_z.values
            ax.scatter(xs, ys, zs, color='orange', s=30)
            for x, y, z, label in zip(xs, ys, zs, objects.class_label.values):
                point = np.array([x, y, z])
                dir_str = pose.world_to_string_direction(point)
                ax.text(x, y, z, f"{label}\n({dir_str})", size=8, color='purple')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Step {step_idx}")
        ax.grid(True)

        # Save to file
        temp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.png"
        plt.tight_layout()
        plt.savefig(temp_path)
        plt.close(fig)

        return str(temp_path)

    

    @staticmethod
    def plot_camera_trajectory_with_step_slider(trajectory_dict, objects_df, floor_plane=None, axis_length=0.05):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        positions = np.array([pose.position for pose in trajectory_dict.values()])
        step_indices = sorted(trajectory_dict.keys())

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        # Plot full trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectory", color='black')

        # Equal axis scaling
        all_points = np.vstack((positions, objects_df[['closest_3d_x', 'closest_3d_y', 'closest_3d_z']].values))
        max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
        mid = all_points.mean(axis=0)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory with Step Slider')
        ax.grid(True)
        

        def view_angles_from_normal(normal):
            """Compute matplotlib view elevation and azimuth from a 3D normal vector."""
            normal = -1 * normal / np.linalg.norm(normal)
            elev_rad = np.arcsin(normal[2])  # angle between normal and XY plane
            azim_rad = np.arctan2(normal[1], normal[0])
            return np.degrees(elev_rad), np.degrees(azim_rad)

        if floor_plane:
            normal, _ = floor_plane
            elev, azim = view_angles_from_normal(normal)
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=30, azim=0)

        # Floor plane visualization
        if floor_plane:
            normal, offset = floor_plane

            # Create a meshgrid around the center area
            plane_size = max_range * 2
            grid_res = 20
            xx, yy = np.meshgrid(
                np.linspace(mid[0] - plane_size, mid[0] + plane_size, grid_res),
                np.linspace(mid[1] - plane_size, mid[1] + plane_size, grid_res)
            )

            # Solve for z using the plane equation: n_x x + n_y y + n_z z + d = 0 => z = (-n_x x - n_y y - d)/n_z
            if abs(normal[2]) > 1e-6:
                zz = (-normal[0] * xx - normal[1] * yy - offset) / normal[2]
                ax.plot_surface(xx, yy, zz, color='cyan', alpha=0.3, edgecolor='none', label='Floor')
            else:
                print("Warning: Floor normal vector too close to horizontal — can't render floor plane.")

        # Plot elements that change per step
        step_pose_plot = ax.scatter([], [], [], color='blue', s=50, label='Camera Position')
        object_plot = ax.scatter([], [], [], color='orange', s=30, label='Objects')
        axis_lines = {'x': None, 'y': None, 'z': None}
        text_labels = []

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        step_slider = Slider(ax_slider, 'Step', step_indices[0], step_indices[-1], valinit=step_indices[0], valstep=1)

        def update(step):
            nonlocal text_labels, axis_lines
            step = int(step)

            for txt in text_labels:
                txt.remove()
            text_labels = []

            pose = trajectory_dict.get(step)
            if pose:
                pos = pose.position
                step_pose_plot._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

                rot = pose.rotation.as_matrix()
                axes = {'x': (rot[:, 0], 'r'), 'y': (rot[:, 1], 'g'), 'z': (rot[:, 2], 'b')}

                for axis in axis_lines.values():
                    if axis:
                        axis.remove()

                for name, (vec, color) in axes.items():
                    end = pos + vec * axis_length
                    axis_lines[name] = ax.plot(
                        [pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                        color=color, linewidth=2
                    )[0]
            else:
                step_pose_plot._offsets3d = ([], [], [])

            frame_objects = objects_df[objects_df.step == step]
            if not frame_objects.empty and pose:
                xs = frame_objects.closest_3d_x.values
                ys = frame_objects.closest_3d_y.values
                zs = frame_objects.closest_3d_z.values
                object_plot._offsets3d = (xs, ys, zs)

                for x, y, z, label in zip(xs, ys, zs, frame_objects.class_label.values):
                    point_world = np.array([x, y, z])
                    direction_str = pose.world_to_string_direction(point_world)
                    text = ax.text(x, y, z, f"{label}\n({direction_str})", size=8, color='purple')
                    text_labels.append(text)
            else:
                object_plot._offsets3d = ([], [], [])

            fig.canvas.draw_idle()


        step_slider.on_changed(update)
        update(step_indices[0])

        plt.tight_layout()
        plt.legend()
        plt.show()





if __name__ == "__main__":
    dm = DataManager()
    dm.load_data(
        #Path(r"C:\Users\nick\OneDrive\Dokumente\Studium\TUM\Master\Semester2\AppliedFoundationModels\work\PathPilot\Data\evaluation\evaluation\outdoor_1\run_279\incremental_analysis_detailed_20250718_232931")
        Path(r"C:\Users\nick\OneDrive\Dokumente\Studium\TUM\Master\Semester2\AppliedFoundationModels\work\PathPilot\Data\run_bonus_stage_two_chairs_and_trash\incremental_analysis_detailed_20250718_224557")
    )

    DataManager.plot_camera_trajectory_with_step_slider(dm.trajectory, dm.objects_df, floor_plane=dm.floor_plane,axis_length=0.5)


