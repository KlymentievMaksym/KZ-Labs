import numpy as np
from PIL import Image
import os
from kernels import *

def rgb_to_halftone(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    halftone = 0.36 * r + 0.53 * g + 0.11 * b
    # halftone = (r + g + b)/3

    return halftone


photo = Image.open("Photos\\test.jpg")

ph_filter = Gauss_11x11()# /7

print(ph_filter)

working_ar = np.array(photo)
working_ar_l = rgb_to_halftone(working_ar).astype(np.uint8)

print(working_ar.shape)
#print(working_ar_l.shape)

filtered_ar = np.zeros_like(working_ar)

ar_size = len(working_ar_l)

if type(ph_filter) == np.ndarray:
    height, width, channels = working_ar.shape
    height_fl, width_fl = ph_filter.shape
    
    height_limit = height_fl - 2
    width_limit = width_fl - 2
    
    for channel in range(channels):
        for row in range(height_limit, height - height_limit):
            for col in range(height_limit, width - height_limit):
                matrix = working_ar[row - height_limit:row + 2, col - width_limit:col + 2, channel]
                filtered_ar[row, col, channel] = np.sum(matrix * ph_filter)# %256
    
    
    
    
    filtered_ar = np.clip(filtered_ar, 0, 255).astype(np.uint8)
    
elif type(ph_filter) == list:
    [x_shift, y_shift] = ph_filter
    filtered_ar[y_shift:, x_shift:] = working_ar[:-y_shift, :-x_shift]
    
photo = Image.fromarray(filtered_ar)

try:
    os.mkdir('Results\\')
except FileExistsError:
    pass

photo.save('Results\\test_result.jpg')