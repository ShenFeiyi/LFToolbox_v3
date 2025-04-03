# -*- coding:utf-8 -*-
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from ImageClass import FullApertureImage, HalfApertureImage
from LightFieldCamera import LFCam

# root = '/Users/feiyishen/Library/CloudStorage/OneDrive-UniversityofArizona/Feiyi_Images_OneDrive/Feiyi_images'
# half_aperture_image_filename = str(11402).zfill(6)+'.tiff'
root = '/Users/feiyishen/Desktop'
half_aperture_image_filename = str(11402).zfill(6)+'.png'
scene_filename = str(11779).zfill(6)+'.png'

SI = FullApertureImage(os.path.join(root, scene_filename))
HAI = HalfApertureImage(os.path.join(root, half_aperture_image_filename))

cam = LFCam(pixel_size = 3.45e-3, EI_shape = (5, 7))

cam.half_aperture = HAI
cam.half_aperture.circle_detect_params = {
    'radius': 100, 'param1': 10, 'param2': 0.95
    }
cam.build_grid_model()

cam.scene = SI
cam.scene.EI_segment_params = cam.public_EI_segment_params
corners = cam.scene._convert_ApCenters_to_corners()

# plt.imshow(cam.half_aperture.image, cmap='gray')
plt.imshow(255-cam.scene.image, cmap='gray')
plt.scatter(cam.ApCenters[:,:,0], cam.ApCenters[:,:,1], color='r')
for i in range(4):
    plt.scatter(corners[:,:,i,0], corners[:,:,i,1], marker='x', color=['k', 'c', 'm', 'y'][i])
plt.show()
