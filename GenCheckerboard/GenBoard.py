# -*- coding:utf-8 -*-
import os
import json
import cv2 as cv
import numpy as np
from PIL import Image

def _genBoard(paperSize, dpi, data):
    info = {}

    res = {'inch': dpi} # dots / inch
    imgSize = {'inch': np.array(paperSize)}
    imgSize['mm'] = imgSize['inch']*25.4
    res['mm'] = res['inch']/25.4 # dots / mm
    imgSize['pixel'] = res['inch'] * imgSize['inch']

    CheckerSize = data['CheckerSize']/data.get('scale',1)
    xoffset = data.get('xoffset', 0)
    yoffset = data.get('yoffset', 0)
    nrows = data['nrows']
    ncols = data['ncols']

    img = 255*np.ones(imgSize['pixel'].astype(int), dtype='uint8')

    checkerSize = {'mm': CheckerSize}
    checkerSize['pixel'] = int(checkerSize['mm']*res['mm'])

    top = np.round((imgSize['pixel'][0] - checkerSize['pixel'] * nrows)//2).astype(int)
    left = np.round((imgSize['pixel'][1] - checkerSize['pixel'] * ncols)//2).astype(int)
    bottom = imgSize['pixel'][0].astype(int) - top
    right = imgSize['pixel'][1].astype(int) - left

    for i, row in enumerate(range(nrows)):
        black = bool(np.mod(i, 2))
        row_pixel = top + row * checkerSize['pixel']
        for j, col in enumerate(range(ncols)):
            col_pixel = left + col * checkerSize['pixel']
            if black:
                img[row_pixel:row_pixel+checkerSize['pixel'], col_pixel:col_pixel+checkerSize['pixel']] = 0
            black = not black

    img = cv.putText(
        img, 'Checkerboard | '+str(nrows)+'x'+str(ncols)+' | Checker Size '+str(checkerSize['mm'])+' mm',
        (500, 500), cv.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, cv.LINE_AA
        )
    img = cv.putText(
        img, 'Width: 150 px = '+str(150/res['mm'])+' mm',
        (500, 1000), cv.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 10, cv.LINE_AA
        )
    linewidth = [i+1 for i in range(150)]
    black = True
    for ii, lw in enumerate(linewidth):
        if black:
            img[1200:1800, 500+sum(linewidth[:ii]):500+sum(linewidth[:ii+1])] = 0
        black = not black

    info['img'] = img
    info['top'], info['bottom'], info['left'], info['right'] = top, bottom, left, right
    info['checkerSize'] = checkerSize
    info['xoffset'], info['yoffset'] = xoffset, yoffset
    return info

def genBlankBoard(paperSize, dpi, data):
    info = _genBoard(paperSize, dpi, data)
    img = info['img']
    image = Image.fromarray(img)
    filename = f"Checkerboard_{data['nrows']}x{data['ncols']}_checker_{data['CheckerSize']}mm.tif"
    image.save(filename, dpi=(dpi, dpi))

def genTextBoard(paperSize, dpi, data):
    info = _genBoard(paperSize, dpi, data)
    img = info['img']
    top, bottom, left, right = info['top'], info['bottom'], info['left'], info['right']
    checkerSize = info['checkerSize']
    xoffset, yoffset = info['xoffset'], info['yoffset']

    font = cv.FONT_HERSHEY_SIMPLEX
    for i, row in enumerate(range(top, bottom, checkerSize['pixel'])):
        for j, col in enumerate(range(left, right, checkerSize['pixel'])):
            # add text
            img = cv.putText(
                    img, chr(65+i)+chr(65+j), (col+xoffset, row-yoffset+checkerSize['pixel']),
                    font, data['fontscale'], (0,0,0), data['fontthickness'], cv.LINE_AA
                    )
    image = Image.fromarray(img)
    filename = 'Checkerboard_'
    filename += str(data['nrows'])
    filename += 'x'
    filename += str(data['ncols'])
    filename += '_checker_'
    filename += str(round(checkerSize['mm']))
    filename += 'mm_text.tif'
    image.save(filename, dpi=(dpi, dpi))

if __name__ == '__main__':
    files = os.listdir('.')
    _ = [os.remove(f) for f in files if f.endswith('tif')]

    # printer parameters
    scale = 0.994
    paper = [8.5, 11] # inch
    DPI = 1200

    for filename in '23456789':
        try:
            with open(filename+'.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
            data['scale'] = scale
            genTextBoard(paper, DPI, data)
        except FileNotFoundError:
            pass

    genBlankBoard(paper, DPI, {'CheckerSize':3,'nrows':6,'ncols':7, 'scale':scale})
    genBlankBoard(paper, DPI, {'CheckerSize':20,'nrows':8,'ncols':8, 'scale':scale})
