# -*- coding:utf-8 -*-
import numpy as np

class LFCam:
    def __init__(self, **kwargs):
        """
        Light Field Camera class
        Calibration and Depth Evaluation
        All units are in [mm]
        """
        self.camera_name = kwargs.get('camera_name', 'Light Field Camera')
        self.pixel_size = kwargs.get('pixel_size')

        self.aperture = None # class half_aperture_image

        self.ApCenters = None # aperture centers, (#row, #col, 2)

    def __repr__(self):
        return self.camera_name

    """
    General Methods
    """

    def save(self, **kwargs):
        """
        Save the camera parameters
        """
        pass

    def load(self, **kwargs):
        """
        Load the camera parameters
        """
        pass

    def _distortion_model(self, **kwargs):
        """
        Distortion model
        """
        pass

    def _sphere_model(self, **kwargs):
        """
        Sphere model
        """
        pass

    def _ray_bundle_waist_eval(self, **kwargs):
        """
        Evaluate  position of ray bundle waist
        """
        pass

    """
    Calibration Methods
    """

    def build_grid_model(self, **kwargs):
        """Build grid model from self half aperture image
        Note:
            May get OptimizeWarning, because only two points are used to fit the line
            To ignore, set ignoreWarning = True
        Args:
            approxRadius (int): Approximate radius of the aperture circle (px)
            imageBoundary (int): Image boundary (px)
            debugDisplay (bool): Display debug information
            ignoreWarning (bool): Ignore warnings
        Returns:
            rotate_matrix (numpy.ndarray): Rotation matrix, (2x2), from rigid grid to tilted grid
            ApCenters (numpy.ndarray): Aperture centers, (#row, #col, 2)
        """
        approxRadius = kwargs.get('approxRadius', 200)  # px
        imageBoundary = kwargs.get('imageBoundary', 50)  # px
        debugDisplay = kwargs.get('debugDisplay', False)  # bool
        ignoreWarning = kwargs.get('ignoreWarning', False)  # bool

        if ignoreWarning:
            import warnings
            warnings.filterwarnings('ignore')

        assert self.aperture is not None, "Aperture image not loaded"

        x_index, y_index = 0, 1
        row_index, col_index = 0, 1

        def line(x, k=1, b=0):
            # assuming no perfect vertical line
            return k * x + b

        if len(self.aperture.shape) > 2:
            # RGB 2 gray
            self.aperture = cv.cvtColor(self.aperture, cv.COLOR_RGB2GRAY)

        # set circle detection parameters outside before calling
        # self.aperture.circle_detect_params = {'radius': approxRadius, 'param1': 10, 'param2': 0.95}
        self.ApCenters = self.aperture.featPoints[:, :, :-1]
        _, self.rotate_matrix = self.reorder_points(self.ApCenters)

        return self.rotate_matrix, self.ApCenters

    def calibrate(self, **kwargs):
        """
        Calibrate the camera, start all the calibration process
        """
        pass

    """
    Depth Evaluation Methods
    """

    def evaluate_depth(self, **kwargs):
        """
        Evaluate the depth of the scene
        """
        pass
