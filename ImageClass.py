# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

class LFImage:
    """Raw Light Field Image class
    """
    def __init__(self, name, local_path):
        """
        Args:
            name (str): Image name
            local_path (str): Local path
        """
        self.name = str(name)
        self.image = cv.imread(local_path, cv.IMREAD_GRAYSCALE)
        self.path = local_path

        if self.image is None:
            raise ValueError(f"Image not found at {local_path}")

        self.elemental_images = None
        self.EI_segment_params = {}

        self.feature_points = None
        self.circle_detect_params = {}

    def __repr__(self):
        return self.name + ' @ ' + self.path

    @property
    def EIs(self, invM, ApCenters):
        if self.elemental_images is None:
            self.elemental_images = self._segment_elemental_images()
        return self.elemental_images

    @property
    def featPoints(self):
        if self.feature_points is None:
            self.feature_points = self._detect_circles()
        return self.feature_points

    def _convert_ApCenters_to_corners(self):
        """convert aperture centers to four corners
        Args:
            self.EI_segment_params (dict): Parameters for segmenting elemental images
                rotation_matrix (numpy.ndarray): 2x2 transformation matrix (rotation & shear)
                ApCenters (numpy.ndarray): Aperture centers, (#row, #col, 2)
        Returns:
            corners (numpy.ndarray): Four corners of the aperture, (#row, #col, 4, 2)
        """
        x_index, y_index = 0, 1
        rotation_matrix = self.EI_segment_params.get('rotation_matrix')
        ApCenters = self.EI_segment_params.get('ApCenters')
        # Convert to corners (upperLeft, upperRight, lowerRight, lowerLeft)
        corners = np.zeros((ApCenters.shape[0], ApCenters.shape[1], 4, 2), dtype=np.float32)
        # x-direction and y-direction unit vectors
        x_dir = rotation_matrix[x_index, :]
        y_dir = rotation_matrix[y_index, :]
        # average length between aperture centers in x and y direction
        x_length = (ApCenters[:, 1:, x_index] - ApCenters[:, :-1, x_index]).mean()
        y_length = (ApCenters[1:, :, y_index] - ApCenters[:-1, :, y_index]).mean()
        # step vectors
        x_step = x_length * x_dir
        y_step = y_length * y_dir
        # upper left corner
        corners[:, :, 0, :] = ApCenters - 0.5 * x_step - 0.5 * y_step
        # upper right corner
        corners[:, :, 1, :] = ApCenters + 0.5 * x_step - 0.5 * y_step
        # lower right corner
        corners[:, :, 2, :] = ApCenters + 0.5 * x_step + 0.5 * y_step
        # lower left corner
        corners[:, :, 3, :] = ApCenters - 0.5 * x_step + 0.5 * y_step
        return corners

    def _segment_elemental_images(self):
        """Segment elemental images
        Args:
            self.EI_segment_params (dict): Parameters for segmenting elemental images
                rotation_matrix (numpy.ndarray): 2x2 transformation matrix (rotation & shear)
                ApCenters (numpy.ndarray): Aperture centers, (#row, #col, 2)
        Returns:
            EIs (dict): Elemental images {#row:{#col:image (Width,Height,Color)}}
            anchors (dict): Five points in each EI {#row:{#col:{points}}}
                points = {'center':p0, 'UL':p_upperLeft, 'UR':p_upperRight, 'LR':p_lowerRight, 'LL':p_lowerLeft}
        """
        pass

    def _reorder_points(self, points, grid_shape):
        """Rearrange a list of points into a rectangular grid
        The points are in a rotated rectangular grid, but their order is not guaranteed.
        The function rearranges them into a rectangular grid with the specified shape.
        Args:
            points (numpy.ndarray): Points to be arranged, shape (#points, 2)
            grid_shape (tuple): Shape of the grid (#rows, #cols)
        Returns:
            reordered_points (numpy.ndarray): Rearranged points, shape (#rows, #cols, 2)
            rotation_matrix (numpy.ndarray): Rotation matrix, from grid to tilted grid, apply its inverse to get grid
        """
        x_index, y_index = 0, 1
        points = np.array(points, dtype=np.float32)
        # PCA using OpenCV to find rotation angle
        mean, eigenvectors = cv.PCACompute(points, mean=None)
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
        # Rotate points to align grid
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        rotated_points = points @ rotation_matrix.T
        # Sort points by Y first (rows), then X (columns)
        sorted_idx = np.lexsort((rotated_points[:, x_index], rotated_points[:, y_index]))
        sorted_points = rotated_points[sorted_idx]
        # Reshape to grid and then sort each row by X
        grid = sorted_points.reshape(grid_shape[0], grid_shape[1], 2)
        for row in grid:
            row[:] = row[row[:, 0].argsort()]
        # Reverse rotation to original orientation
        reordered_points = grid.reshape(-1, 2) @ rotation_matrix
        reordered_points = reordered_points.reshape(grid_shape[0], grid_shape[1], 2)
        return reordered_points, rotation_matrix

    def reorder_points(self, grid_shape):
        """For public calling, reorder points to a rectangular grid"""
        return self._reorder_points(self.featPoints, grid_shape)

    def _detect_circles(self):
        """Detect circles using the Hough Circle Transform
        Args:
            self.circle_detect_params (dict): Parameters for Hough Circle Transform
                radius (float): Radius of the circles to be detected
                dp (float): Inverse ratio of the accumulator resolution to the image resolution
                minDist (float): Minimum distance between detected centers
                param1 (int): First method-specific parameter
                param2 (int): Second method-specific parameter
                minRadius (float): Minimum circle radius
                maxRadius (float): Maximum circle radius
        Returns:
            points (numpy.ndarray): Detected circles, shape (#circles, 3)
                points[i] = (x, y, r) where (x, y) is the center and r is the radius
        """
        radius = self.circle_detect_params.get('radius', 10)
        dp = self.circle_detect_params.get('dp', 1.5)
        minDist = self.circle_detect_params.get('minDist', 10)
        param1 = self.circle_detect_params.get('param1', 10)
        param2 = self.circle_detect_params.get('param2', 0.95)
        minRadius = self.circle_detect_params.get('minRadius', 0.5)
        maxRadius = self.circle_detect_params.get('maxRadius', 1.5)
        """HoughCircles returns (x, y) points"""
        circles = cv.HoughCircles(
            self.image, cv.HOUGH_GRADIENT_ALT, dp=dp,
            minDist=minDist, param1=param1, param2=param2,
            minRadius=int(radius*minRadius), maxRadius=int(radius*maxRadius)
            )
        points = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # i => (x, y, r)
                p = np.array([i[0], i[1]])
                points.append(p)
        points = np.array(points)
        # remove duplicate points
        points = np.unique(points, axis=0)
        return points

class FullApertureImage(LFImage):
    """Full Aperture Image class
    """
    def __init__(self, local_path):
        """
        Args:
            local_path (str): Local path
        """
        super().__init__('Full aperture image', local_path)

class HalfApertureImage(LFImage):
    """Half Aperture Image class
    """
    def __init__(self, local_path):
        """
        Args:
            local_path (str): Local path
        """
        super().__init__('Half aperture image', local_path)

class DotSampleImage(LFImage):
    """Dot Sample Image class
    """
    def __init__(self, name, local_path, posx, posy, posz):
        """
        Args:
            name (str): Image name
            local_path (str): Local path
            posx (float): Position x
            posy (float): Position y
            posz (float): Position z
        """
        super().__init__(name, local_path)
        self.posx = posx
        self.posy = posy
        self.posz = posz

class STSampleImage(DotSampleImage):
    """ST Sample Image class
    """
    def __init__(self, local_path, posx, posy):
        """
        Args:
            local_path (str): Local path
            posx (float): Position x
            posy (float): Position y
        """
        name = 'ST Sample @ ({:.1f}, {:.1f})'.format(posx, posy)
        super().__init__(name, local_path, posx, posy, 0)

class ObjSampleImage(DotSampleImage):
    """Object Sample Image class
    """
    def __init__(self, local_path, posx, posy, posz):
        """
        Args:
            local_path (str): Local path
            posx (float): Position x
            posy (float): Position y
            posz (float): Position z
        """
        name = 'Object Sample @ ({:.1f}, {:.1f}, {:.1f})'.format(posx, posy, posz)
        super().__init__(name, local_path, posx, posy, posz)
