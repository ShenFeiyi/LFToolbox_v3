# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

def reorder_points(points):


    x_index, y_index = 0, 1

    row_index, col_index = 0, 1



    points = np.array(points, dtype=np.float32)

    # PCA using OpenCV to find rotation angle

    mean, eigenvectors = cv2.PCACompute(points, mean=None)

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



    return reordered_points, rotation_matrix


def convert_ApCenters_to_corners(rotate_matrix, ApCenters):

    x_index, y_index = 0, 1

    row_index, col_index = 0, 1


    # Convert to corners (upperLeft, upperRight, lowerRight, lowerLeft)

    corners = np.zeros((ApCenters.shape[0], ApCenters.shape[1], 4, 2), dtype=np.float32)

    # size of the grid

    x_size = (ApCenters[:, 1:, x_index] - ApCenters[:, :-1, x_index]).mean()

    y_size = (ApCenters[1:, :, y_index] - ApCenters[:-1, :, y_index]).mean()

    shift = np.ones(ApCenters.shape)
    shift[:,:,0] = x_size/2
    shift[:,:,1] = y_size/2
    # Upper left corner

    corners[:, :, 0, :] = ApCenters - shift
    return corners

# Example usage
points = [
    [0, 0], [1, 1], [3, 2], [2, 2],
    [1, 0], [2, 0], [3, 0], [0, 1],
    [2, 1], [3, 1], [0, 2], [1, 2]
]
points = np.array(points) + np.random.rand(12,2)/20
angle = 25 * (np.pi/180)
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])
points = points @ R

grid_shape = [3, 4]
reordered, rotation_matrix = reorder_points(points)
print(reordered.reshape(grid_shape+[2]))

for i, p in enumerate(reordered):
    plt.scatter(p[0], p[1], marker='x')
    #plt.text(p[0], p[1], str(i), fontsize=22)

corners = reordered @ rotation_matrix
corners = corners.reshape(-1,2)
for i, p in enumerate(corners):
    plt.scatter(p[0], p[1])
    plt.text(p[0], p[1], str(i), fontsize=22)

plt.show()
