import sys
import itertools
import numpy as np
import jax
import abtem
import dask
import ase
from ase.build import bulk, surface
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import pyqtgraph.opengl as gl
import temgym_core.components as comp
import temgym_core.source as sources
from temgym_ui.window_3d import TemGymWindow3D, LABEL_RADIUS, Z_ORIENT
from temgym_ui.window import GridGeomMixin, GridGeomParams
from matplotlib import colormaps

dask.config.set({"num_workers": 1})
jax.config.update("jax_platform_name", "cpu")
abtem.config.set({"device": "gpu"})
abtem.config.set({"dask.chunk-size-gpu": "2048 MB"})


global ATOMS_GEOMETRY
ATOMS_GEOMETRY = []


def make_sphere(x, y, z, radius, color):
    md = gl.MeshData.sphere(rows=20, cols=20, radius=radius)
    colors = np.ones((md.faceCount(), 4), dtype=float)
    colors[:, :3] = np.asarray(color)[np.newaxis, :]
    md.setFaceColors(colors)
    m3 = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded', glOptions='opaque')
    m3.translate(x, y, z)
    return m3


class DetectorGUIWrapper(GridGeomMixin):
    def __init__(self, component: comp.Detector):
        self.detector = component

    def _get_extents(self) -> GridGeomParams:
        # (cx, cy, w, h, rotation, z)
        h, w = self.detector.shape
        scale = min(self.detector.pixel_size)
        h *= scale
        w *= scale
        cx, cy = self.detector.centre
        rotation = self.detector.rotation
        z = self.detector.z
        return GridGeomParams(
            w, h, cx, cy, np.deg2rad(rotation), z, self.detector.shape,
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


def make_model(theta, radius: float = 0.075):
    offset = (radius * np.cos(theta), radius * np.sin(theta))
    scan_thickness = 0.1
    return (
        (source := sources.ParallelBeam(0., 0.01)),
        (scanner := MagicScanPrecessor(source.z + 0.6, scan_thickness, offset)),
        (input_lens := comp.Lens(scanner.z + scanner.thickness, 0.4)),
        (atoms := Atoms(input_lens.z + input_lens.focal_length)),
        (output_lens := comp.Lens(atoms.z + input_lens.focal_length, input_lens.focal_length)),
        (descanner := MagicDeScanPrecessor(output_lens.z, scan_thickness, offset)),
        (DetectorWithGUI(descanner.z + 0.6, (0.02,) * 2, (32,) * 2)),
    )


def show(model, num_rays: int = 256, animate: bool = True, theta_seq=None, images=None):
    AppWindow = QApplication(sys.argv)
    viewer = TemGymWindow3D(num_rays=num_rays)
    viewer.set_model(model)

    timer = QTimer(viewer)
    timer.setInterval(75)

    cmap = colormaps["magma"].resampled(256)

    if animate:
        default_num = 64
        if theta_seq is None:
            theta_seq = np.linspace(0., 2 * np.pi, num=default_num)
        theta_iter = itertools.cycle(theta_seq)

        if images is None:
            images = np.random.uniform(size=(default_num, 8, 8))
        images_iter = itertools.cycle(images)

        def iterate():
            theta = next(theta_iter)
            viewer.set_model(make_model(theta), geometry=False, camera=False, tree=False)
            image_2d = next(images_iter)
            data = (cmap(image_2d.ravel()).reshape(*image_2d.shape, 4) * 255).astype(np.uint8)
            viewer.detector_image_geom.setData(data)

            idx = int(np.round(np.rad2deg(theta)))
            viewer.tem_window.grabFramebuffer().save(f'images/img{idx:>04d}.png')

        timer.timeout.connect(iterate)
        timer.start()

    viewer.show()
    AppWindow.exec()


def precession_tilts(
    precession_angle: float,
    num_samples: int,
    min_azimuth: float = 0.0,
    max_azimuth: float = 2 * np.pi,
    endpoint: bool = False,
):
    azimuthal_angles = np.linspace(
        min_azimuth, max_azimuth, num=num_samples, endpoint=endpoint
    )

    tilt_x = precession_angle * np.cos(azimuthal_angles)
    tilt_y = precession_angle * np.sin(azimuthal_angles)

    return np.array([tilt_x, tilt_y], dtype=float).T, azimuthal_angles


if __name__ == "__main__":
    model = make_model(0.)
    atoms_obj = model[3]
    assert isinstance(atoms_obj, Atoms)

    silicon = bulk("Si", crystalstructure="diamond")
    silicon_111 = surface(
        silicon, (1, 1, 1), layers=1, periodic=True
    )
    # atoms = silicon_111 * (5, 7, 1)

    srtio3 = ase.io.read("SrTiO3.cif")
    atoms = srtio3 * (4, 4, 1)

    colors = {
        "Sr": np.asarray((155, 224, 36)) / 255,
        "Ti": np.asarray((36, 215, 224)) / 255,
        "O": np.asarray((224, 36, 64)) / 255,
        "Si": np.asarray((224, 161, 36)) / 255,
    }

    radii = {
        "Sr": 0.013,
        "Ti": 0.008,
        "O": 0.01,
        "Si": 0.01,
    }

    import pathlib
    _ = tuple(fp.unlink() for fp in pathlib.Path("./images").iterdir())

    atoms.center(about=(0., 0., 0.))
    sf = 50
    for sym, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        ATOMS_GEOMETRY.append(
            make_sphere(x / sf, y / sf, ((z / sf) + atoms_obj.z) * Z_ORIENT, radii[sym], colors[sym])
        )
    if False:
        silicon_111_orthogonal = abtem.orthogonalize_cell(silicon_111)
        sf = 3
        atoms = silicon_111_orthogonal * (13 * sf, 8 * sf, 40)
        frozen_phonons = abtem.FrozenPhonons(
            atoms, 8, sigmas=0.078,
        )
        potential = abtem.Potential(
            frozen_phonons,
            sampling=0.1,
            projection="infinite",
            slice_thickness=2,
        )
        wave = abtem.Probe(semiangle_cutoff=2, energy=100e3)
        wave.grid.match(potential)
        tilts, azimuths = precession_tilts(np.deg2rad(3) * 1000, 64)
        wave.tilt = tilts
        measurement = wave.multislice(potential).diffraction_patterns(max_angle=30).mean(0).compute()
        print(measurement.array.shape)
        images = measurement.array.get()
        images /= images.max(axis=(-1, -2), keepdims=True)
        np.savez("images.npz", azimuths=azimuths, images=images)
    else:
        image_data = np.load("images.npz")
        azimuths = image_data["azimuths"]
        images = image_data["images"]
    show(model, animate=True, theta_seq=azimuths, images=images)
