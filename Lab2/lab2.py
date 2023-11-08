import numpy as np
from PIL import Image
import os
import datetime as dtime
from kernels import *

def rgb_to_halftone(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    halftone = 0.36 * r + 0.53 * g + 0.11 * b
    # halftone = (r + g + b)/3

    return halftone


photo = Image.open("Photos\\test.jpg")

ph_filter = Inversia()# /7

# ph_filter = np.array([
#     [0, 0, 0],
#     [0, -1, 0],
#     [0, 0, 0]
#     ], dtype=np.float16)

print()
print(ph_filter)
print()

working_ar = np.array(photo)
working_ar_l = rgb_to_halftone(working_ar).astype(np.uint8)

print(working_ar.shape)
#print(working_ar_l.shape)



ar_size = len(working_ar_l)

# if type(ph_filter) == np.ndarray:
#     height, width, channels = working_ar.shape
#     height_fl, width_fl = ph_filter.shape
    
#     height_limit = height_fl - 2
#     width_limit = width_fl - 2
    
#     for channel in range(channels):
#         for row in range(height_limit, height - height_limit):
#             for col in range(height_limit, width - height_limit):
#                 matrix = working_ar[row - height_limit:row + 2, col - width_limit:col + 2, channel]
#                 filtered_ar[row, col, channel] = np.sum(matrix * ph_filter)# %256

height, width, channels = working_ar.shape
height_fl, width_fl = ph_filter.shape
width_limit = width_fl - 2
height_limit = height_fl - 2

height_nd = int(height/height_fl)
width_nd = int(width/height_fl)

# print(type(height_nd))

print(working_ar[:height_nd, :width_nd, :].shape)
print()
print(height_nd, width_nd, height/height_nd, width/width_nd)
print()

filtered_ar = np.zeros_like(working_ar[:height_nd, :width_nd, :]) # 

tests = np.split(working_ar, height_nd, 0)

k = 0

for test in tests:
    #print(test.shape)
    test = np.split(test, width_nd, 1)
    # print(test[0].shape)
    tests[k] = test
    k += 1
    
   # np.split(, 3, 1)
    
    # filtered_ar = np.clip(filtered_ar, 0, 255).astype(np.uint8)
tests = np.array(tests)

print(tests.shape)

height, width, i, j, channels = tests.shape

for channel in range(channels):
    for row in range(height):
        for col in range(width):
            #if k == 778:
            #print(tests[row, col, :, :, channel].shape)
            #print(tests[row, col, :, :, channel]*ph_filter)
            
            filtered_ar[row, col, channel] = np.sum(tests[row, col, :, :, channel]*ph_filter)
                # k = 0
            
print(filtered_ar.shape)           
          
photo = Image.fromarray(filtered_ar)

# try:
#     os.mkdir('Results\\')
# except FileExistsError:
#     pass

photo.show()

photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.jpg')