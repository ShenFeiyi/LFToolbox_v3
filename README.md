# Light Field Calibration Toolbox -- version 3.7.0

**Feiyi Shen** 

[toc]

- [0. Introduction](#0-introduction)
- [1. Environment](#1-Environment)
- [2. Definitions](#2-Definitions)
- [3. Method](#3-Method)
- [4. Result](#4-Result)
- [5. Code](#5-Code)
- [6. Log](#6-Log)

------

## 0. Introduction

A [light field camera](https://en.wikipedia.org/wiki/Light_field_camera) calibration toolbox for light field 2.0, also known as resolution priority type light field, or focused light field, focused plenoptic camera. Typically, a light field camera consists of an aperture, a camera lens, a micro-lens array (MLA) and a digital image sensor. 

<p align="center"><img src='./README_IMAGES/cur_setup.png' height=200><img src='./README_IMAGES/housing_cad.png' height=200></p>

<p align="center">Fig. 1. Current Light Field Camera System Setup</p>

The goal of this project is to calibrate the light field camera and accurately estimate the depth of a 3D scene. 



## 1. Environment

- `Python 3.12.6` 



## 2. Definitions

A display panel is used to help with the calibration process. The optical axis is defined by a pinhole on the rail. The origin of the display is the intersection point of the optical axis and the display. Facing the display, the horizontal line from left to right through the display origin is the *x axis*; the vertical line from top to bottom through the display origin is the *y axis*. The optical axis is the *z axis*, and its positive direction is the direction where the camera is facing. The intersection point of optical axis and entrance pupil plane is the origin of the depth. In other words, the entrance pupil is at depth equal to $0mm$. 

<p align="center"><img src='./README_IMAGES/display.png' height=200><img src='./README_IMAGES/coordinate_definition.jpeg' height=200></p>

<p align="center">Fig. 2. Display Panel & Coordinate definition</p>



## 3. Method



## 4. Result



## 5. Code



## 6. Log

**Version 3.7.0** 

**2025/03/17** Upload `README.md`. Currently cleaning up the code and redoing the calibration. The reason is that, last year, a new project came and I have to postpone this project. Now I'm trying to finish this project as soon as possible. 
