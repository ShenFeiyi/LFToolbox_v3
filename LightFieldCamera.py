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
        self.EI_shape = kwargs.get('EI_shape')  # (#row, #col)

        # images
        self.scene = None
        self.full_aperture = None # class full_aperture_image
        self.half_aperture = None # class half_aperture_image
        self.ST_sample_images = None # list of class ST_sample_images
        self.obj_sample_images = None # list of class obj_sample_images

        # feature points
        self.ApCenters = None # aperture centers, (#row, #col, 2)
        self.ST_samples = None
        self.obj_samples = None

        # parameters
        self.public_EI_segment_params = None
        self.rotation_matrix = None
        self.ST_mapping_params = None
        self.UV_rough_location = None
        self.UV_params = None

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
            ApCenters (numpy.ndarray): Aperture centers, (#row, #col, 2)
            rotation_matrix (numpy.ndarray): Rotation matrix, (2x2)
                if image is clockwise, then corresponding angle < 0, counterclockwise angle > 0
                to rotate the image to the original position, use ** rotation_matrix.T **
        """
        approxRadius = kwargs.get('approxRadius', 200)  # px
        imageBoundary = kwargs.get('imageBoundary', 50)  # px
        debugDisplay = kwargs.get('debugDisplay', False)  # bool
        ignoreWarning = kwargs.get('ignoreWarning', False)  # bool

        if ignoreWarning:
            import warnings
            warnings.filterwarnings('ignore')

        assert self.half_aperture is not None, "Aperture image not loaded"

        x_index, y_index = 0, 1
        row_index, col_index = 0, 1

        if len(self.half_aperture.image.shape) > 2:
            # RGB 2 gray
            self.half_aperture.image = cv.cvtColor(self.half_aperture.image, cv.COLOR_RGB2GRAY)

        # set circle detection parameters outside before calling
        # self.half_aperture.circle_detect_params = {'radius': approxRadius, 'param1': 10, 'param2': 0.95}
        ApCenters_detected = self.half_aperture.featPoints # (#total_points, 2)
        # remove points too close to the image boundary
        image_size = self.half_aperture.image.shape[:2]
        x_size, y_size = image_size[col_index], image_size[row_index]
        ApCenters_detected = ApCenters_detected[ApCenters_detected[:, x_index] > imageBoundary]
        ApCenters_detected = ApCenters_detected[ApCenters_detected[:, x_index] < x_size - imageBoundary]
        ApCenters_detected = ApCenters_detected[ApCenters_detected[:, y_index] > imageBoundary]
        ApCenters_detected = ApCenters_detected[ApCenters_detected[:, y_index] < y_size - imageBoundary]
        # update feature points
        self.half_aperture.feature_points = ApCenters_detected
        # reshape into (#row, #col, 2)
        self.ApCenters, self.rotation_matrix = self.half_aperture.reorder_points(self.EI_shape)
        self.public_EI_segment_params = {'rotation_matrix': self.rotation_matrix, 'ApCenters': self.ApCenters}
        return self.ApCenters, self.rotation_matrix

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
