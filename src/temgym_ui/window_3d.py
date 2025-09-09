from PySide6.QtGui import QVector3D
from PySide6.QtCore import (
    Slot,
)
from PySide6.QtWidgets import (
    QMainWindow,
)

import pyqtgraph.opengl as gl

import numpy as np

from .utils import as_gl_lines

from temgym_core.ray import Ray
from temgym_core.run import solve_model, run_iter
from temgym_core.transfer import transfer_rays
from .window import GUIWrapper


LABEL_RADIUS = 0.3
Z_ORIENT = -1
RAY_COLOR = (0.0, 0.8, 0.0)
XYZ_SCALING = np.asarray((1, 1, 1.0))
LENGTHSCALING = 1
MRAD = 1e-3
UPDATE_RATE = 100
BKG_COLOR_3D = (0, 0, 0, 255)


class TemGymWindow3D(QMainWindow):
    """
    Create the UI Window
    """

    def __init__(self, *args, num_rays: int = 64, **kwargs):
        """Init important parameters

        Parameters
        ----------
        model : class
            Microscope model
        """
        super().__init__(*args, **kwargs)
        self.num_rays = num_rays
        self._model = None

        # Set some main window's properties
        self.setWindowTitle("TemGym")
        self.resize(800, 600)

        # Create the display and the buttons
        self.create3DDisplay()
        self.setCentralWidget(self.tem_window)

    def set_model(
        self, model, tree: bool = True, geometry: bool = True, camera: bool = True
    ):
        self._model = model
        if geometry:
            self.add_geometry(model)
        if camera:
            self.update_camera(model)
        self.update_rays(model, self.num_rays)

    def add_geometry(self, model):
        self.tem_window.clear()
        # Loop through all of the model model
        # and add their geometry to the TEM window.
        # FIXME Add in reverse to simulate better depth stacking
        wrappers = []
        for component in model:
            try:
                wrapper = component.gui()
            except AttributeError:
                wrapper = GUIWrapper
            wrappers.append(wrapper)
        wrappers = list(reversed(wrappers))
        for wrapper, component in zip(wrappers, model):
            for geometry in reversed(wrapper.geometry(component)):
                self.tem_window.addItem(geometry)
        # Add labels next so they appear above geometry
        for wrapper, component in zip(wrappers, model):
            self.tem_window.addItem(wrapper.label(component))
        # Add the ray geometry last so it is always on top
        self.tem_window.addItem(self.ray_geometry)

    def update_camera(self, components):
        z_vals = tuple(c.z for c in components)
        mid_z = (min(z_vals) + max(z_vals)) / 2.0
        mid_z *= Z_ORIENT
        xyoffset = (0.2 * mid_z, -0.2 * mid_z)
        self.tem_window.setCameraParams(center=QVector3D(*xyoffset, mid_z))

    @Slot()
    def update_rays(self, model, num_rays: int):
        optical_axis_ray = Ray(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        transfer_matrices = solve_model(optical_axis_ray, model)
        transfer_matrices = transfer_matrices[1:]
        z_vals = np.asarray(
            tuple(ray.z for _, ray in run_iter(optical_axis_ray, model))
        )
        z_vals = z_vals[1:]
        input_rays = model[0].generate_array(num_rays, random=False)

        xy_coords = transfer_rays(input_rays, transfer_matrices)

        vertices = as_gl_lines(xy_coords, z_vals, z_mult=Z_ORIENT)
        self.ray_geometry.setData(
            pos=vertices * XYZ_SCALING,
            color=RAY_COLOR + (0.05,),
        )

    def create3DDisplay(self):
        """Create the 3D Display"""
        # Create the 3D TEM Widnow, and plot the components in 3D
        self.tem_window = gl.GLViewWidget()
        self.tem_window.setBackgroundColor(BKG_COLOR_3D)

        # Get the model mean height to centre the camera origin
        mean_z = 0.0
        mean_z *= Z_ORIENT

        xyoffset = (0.2 * mean_z, -0.2 * mean_z)
        # Define Camera Parameters
        initial_camera_params = {
            "center": QVector3D(*xyoffset, mean_z),
            "fov": 35,
            "azimuth": 45.0,
            "distance": 3.5 * abs(mean_z),
            "elevation": 25.0,
        }
        self.tem_window.setCameraParams(**initial_camera_params)

        self.ray_geometry = gl.GLLinePlotItem(mode="lines", width=2)

        # Add the window to the dock
        return self.tem_window
