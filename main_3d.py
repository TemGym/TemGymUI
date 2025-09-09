import sys
import numpy as np
import jax
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import pyqtgraph.opengl as gl
import temgym_core.components as comp
import temgym_core.source as sources
from temgym_ui.window_3d import TemGymWindow3D, LABEL_RADIUS, Z_ORIENT, Ray, solve_model
from temgym_ui.window import GridGeomMixin, GridGeomParams

jax.config.update("jax_platform_name", "cpu")


class DetectorGUIWrapper(GridGeomMixin):
    def __init__(self, component: comp.Detector):
        self.detector = component

    def _get_extents(self) -> GridGeomParams:
        # (cx, cy, w, h, rotation, z)
        h, w = self.detector.shape
        scale = 0.005
        h *= scale
        w *= scale
        cx, cy = self.detector.centre
        rotation = self.detector.rotation
        z = self.detector.z
        return GridGeomParams(
            w, h, cx, cy, np.deg2rad(rotation), z, (32, 32),
        )

    def geometry(self, component: "DetectorWithGUI"):
        return self.get_geom()

    @staticmethod
    def label(component) -> gl.GLTextItem:
        return gl.GLTextItem(
            pos=np.array([-LABEL_RADIUS, LABEL_RADIUS, Z_ORIENT * component.z]),
            text=type(component).__name__,
            color="w",
        )


class DetectorWithGUI(comp.Detector, GridGeomMixin):
    def gui(self):
        return DetectorGUIWrapper(self)


@comp.jdc.pytree_dataclass
class MagicDeScanPrecessor:
    z: float
    thickness: float
    offset: tuple[float, float]

    def __call__(self, ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        shift_x = ray._one * self.offset[0]
        shift_y = ray._one * self.offset[1]
        return ray.derive(
            x=x - shift_x,
            y=y - shift_y,
            pathlength=ray.pathlength + dx * x + dy * y,
        )


def make_model(theta, radius: float = 0.06):
    offset = (radius * np.cos(theta), radius * np.sin(theta))
    return (
        (source := sources.ParallelBeam(0., 0.01, offset_xy=offset)),
        (input_lens := comp.Lens(0.5, 0.1)),
        (
            scanner := DetectorWithGUI(
                input_lens.z + input_lens.focal_length, (0.001,) * 2, (32, 32)
            )
        ),
        # (output_lens := comp.Lens(scanner.z + input_lens.focal_length, input_lens.focal_length)),
        (descanner := MagicDeScanPrecessor(scanner.z + input_lens.focal_length, 0.05, offset)),
        (detector := DetectorWithGUI(descanner.z + 0.5, (0.001,) * 2, (128, 128))),
    )


def show(model, num_rays: int = 64, animate: bool = True):
    AppWindow = QApplication(sys.argv)
    viewer = TemGymWindow3D(num_rays=num_rays)
    viewer.set_model(model)

    timer = QTimer(viewer)
    timer.setInterval(50)

    if animate:
        theta = 0.

        def iterate():
            nonlocal theta
            viewer.set_model(make_model(theta), camera=False, tree=False)
            theta += 0.1

        timer.timeout.connect(iterate)
        timer.start()

    # viewer.show()
    # AppWindow.exec()


if __name__ == "__main__":
    model = make_model(0.)
    print(model)
    optical_axis_ray = Ray(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    transfer_matrices = solve_model(optical_axis_ray, model)
    # show(model, animate=True)
    pass
