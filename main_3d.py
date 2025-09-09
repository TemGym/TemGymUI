import sys
import numpy as np
import jax
from ase.build import bulk, surface
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import pyqtgraph.opengl as gl
import temgym_core.components as comp
import temgym_core.source as sources
from temgym_ui.window_3d import TemGymWindow3D, LABEL_RADIUS, Z_ORIENT
from temgym_ui.window import GridGeomMixin, GridGeomParams
from matplotlib import colormaps

jax.config.update("jax_platform_name", "cpu")


global ATOMS_GEOMETRY
ATOMS_GEOMETRY = []


def make_sphere(x, y, z, radius):
    md = gl.MeshData.sphere(rows=10, cols=20, radius=radius)
    colors = np.ones((md.faceCount(), 4), dtype=float)
    colors[:, 0] = 0.7
    colors[:, 1] = 0.5
    colors[:, 2] = 0.2
    md.setFaceColors(colors)
    m3 = gl.GLMeshItem(meshdata=md, smooth=False)
    m3.translate(x, y, z)
    return m3


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


class AtomsGUIWrapper:
    def __init__(self, component: "Atoms"):
        self.atoms = component

    def geometry(self, component: "Atoms"):
        return ATOMS_GEOMETRY


@comp.jdc.pytree_dataclass
class Atoms(comp.Component):
    z: float

    def __call__(self, ray):
        return ray

    def gui(self):
        return AtomsGUIWrapper(self)


class DetectorWithGUI(comp.Detector, GridGeomMixin):
    def gui(self):
        return DetectorGUIWrapper(self)


@comp.jdc.pytree_dataclass
class MagicScanPrecessor(comp.Component):
    z: float
    thickness: float
    offset: tuple[float, float]

    def __call__(self, ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        shift_x = self.offset[0]
        shift_y = self.offset[1]
        return ray.derive(
            x=x - shift_x,
            y=y - shift_y,
            pathlength=ray.pathlength + dx * x + dy * y,
            z=ray.z + self.thickness,
        )


@comp.jdc.pytree_dataclass
class MagicDeScanPrecessor(comp.Component):
    z: float
    thickness: float
    offset: tuple[float, float]

    def __call__(self, ray):
        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        shift_x = self.offset[0]
        shift_y = self.offset[1]
        return ray.derive(
            x=x - shift_x,
            y=y - shift_y,
            pathlength=ray.pathlength + dx * x + dy * y,
            z=ray.z + self.thickness,
        )


def make_model(theta, radius: float = 0.15):
    offset = (radius * np.cos(theta), radius * np.sin(theta))
    scan_thickness = 0.1
    return (
        (source := sources.ParallelBeam(0., 0.01)),
        (scanner := MagicScanPrecessor(source.z + 0.4, scan_thickness, offset)),
        (input_lens := comp.Lens(scanner.z + scanner.thickness, 0.25)),
        (atoms := Atoms(input_lens.z + input_lens.focal_length)),
        (output_lens := comp.Lens(atoms.z + input_lens.focal_length, input_lens.focal_length)),
        (descanner := MagicDeScanPrecessor(output_lens.z, scan_thickness, offset)),
        (DetectorWithGUI(descanner.z + 0.5, (0.001,) * 2, (128, 128))),
    )


def show(model, num_rays: int = 64, animate: bool = True):
    AppWindow = QApplication(sys.argv)
    viewer = TemGymWindow3D(num_rays=num_rays)
    viewer.set_model(model)

    timer = QTimer(viewer)
    timer.setInterval(50)

    cmap = colormaps["viridis"].resampled(256)

    if animate:
        theta = 0.

        def iterate():
            nonlocal theta
            viewer.set_model(make_model(theta), geometry=False, camera=False, tree=False)
            theta += 0.1

            image_2d = np.random.uniform(size=(8, 8))
            data = (cmap(image_2d.ravel()).reshape(*image_2d.shape, 4) * 255).astype(np.uint8)
            viewer.detector_image_geom.setData(data)

        timer.timeout.connect(iterate)
        timer.start()

    viewer.show()
    AppWindow.exec()


if __name__ == "__main__":
    model = make_model(0.)
    atoms_obj = model[3]
    assert isinstance(atoms_obj, Atoms)

    atoms_obj.z
    silicon = bulk("Si", crystalstructure="diamond")
    silicon_111 = surface(
        silicon, (1, 1, 1), layers=1, periodic=True
    )
    atoms = silicon_111 * (5, 7, 1)
    atoms.center(about=(0., 0., 0.))
    sf = 50
    for x, y, z in atoms.get_positions():
        ATOMS_GEOMETRY.append(
            make_sphere(x / sf, y / sf, ((z / sf) + atoms_obj.z) * Z_ORIENT, 0.01)
        )

    show(model, animate=True)
