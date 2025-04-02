# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

def reorder_points(points, grid_rows):
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

    # Cluster points into rows using k-means
    _, labels, centers = cv2.kmeans(rotated_points[:,1].reshape(-1,1), grid_rows, None,
                                    criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10000, 0.001),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    rows = [[] for _ in range(grid_rows)]
    for label, point in zip(labels.flatten(), rotated_points):
        rows[label].append(point)

    # Sort rows vertically by their mean Y values
    rows.sort(key=lambda r: np.mean([p[1] for p in r]))

    # Sort each row horizontally by X
    sorted_points = []
    for row in rows:
        row_sorted = sorted(row, key=lambda p: p[0])
        sorted_points.extend(row_sorted)

    sorted_points = np.array(sorted_points)

    # Shift points to upper left by half grid size
    grid_center = (np.max(sorted_points, axis=0) + np.min(sorted_points, axis=0)) / 2
    sorted_points -= grid_center

    # Reverse rotation to original orientation
    reordered_points = sorted_points @ rotation_matrix

    return reordered_points

# Example usage
points = [
    [4 + np.random.rand(), 4 + np.random.rand()], [1 + np.random.rand(), 1 + np.random.rand()], [2 + np.random.rand(), 2 + np.random.rand()], [3 + np.random.rand(), 3 + np.random.rand()],
    [4 + np.random.rand(), 1 + np.random.rand()], [1 + np.random.rand(), 4 + np.random.rand()], [2 + np.random.rand(), 3 + np.random.rand()], [3 + np.random.rand(), 2 + np.random.rand()],
    [3 + np.random.rand(), 1 + np.random.rand()], [2 + np.random.rand(), 4 + np.random.rand()], [1 + np.random.rand(), 3 + np.random.rand()], [4 + np.random.rand(), 2 + np.random.rand()],
    [5 + np.random.rand(), 5 + np.random.rand()], [5 + np.random.rand(), 2 + np.random.rand()], [5 + np.random.rand(), 3 + np.random.rand()], [5 + np.random.rand(), 1 + np.random.rand()]
]

grid_rows = 4
reordered = reorder_points(points, grid_rows)
print(reordered)
