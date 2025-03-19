# -*- coding:utf-8 -*-

class LFCam:
    def __init__(self, **kwargs):
        """
        Light Field Camera class
        Calibration and Depth Evaluation
        All units are in [mm]
        """
        self.camera_name = kwargs.get('camera_name', 'Light Field Camera')

        self.pixel_size = kwargs.get('pixel_size')

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

    def _detect_circles(self, **kwargs):
        """
        Detect circles in the image
        """
        pass

    def _segment_elemental_images(self, **kwargs):
        """
        Segment elemental images
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
