import numpy as np
from PIL import Image
import os


def rgb_to_halftone(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    halftone = 0.36 * r + 0.53 * g + 0.11 * b
    # halftone = (r + g + b)/3

    return halftone


photo = Image.open("Photos\\test.jpg")

ph_filter = np.array([
    [ 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1 ],
    [ 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0 ],
], dtype=np.float32)/7

working_ar = np.array(photo)
working_ar_l = rgb_to_halftone(working_ar).astype(np.uint8)
print(working_ar.shape)
print(working_ar_l.shape)

filtered_ar = np.zeros_like(working_ar)

ar_size = len(working_ar_l)

height, width, channels = working_ar.shape
height_fl, width_fl = ph_filter.shape
# print(working_ar[..., 2])

# for i in range (0, 3):
#     for j in range(0, 3):
#         print(i, j, ph_filter[i][j])
#         # print(ph_filter[i][j])

height_limit = height_fl - 2
width_limit = width_fl - 2

for channel in range(channels):
    for row in range(height_limit, height - height_limit):
        for col in range(height_limit, width - height_limit):
            matrix = working_ar[row - height_limit:row + 2, col - width_limit:col + 2, channel]
            filtered_ar[row, col, channel] = np.sum(matrix * ph_filter)# %256


    #         sum_by_filter = 0
#         for i in range (0, 3):
#             for j in range(0, 3):
#                 sum_by_filter += working_ar_l[(row + i - 1) % ar_size][(col + j - 1) % ar_size] * ph_filter[i][j]
#         filtered_ar[row][col] = sum_by_filter

# for i in range(1, len(working_ar_l) - 1):
#     for j in range(1, len(working_ar_l[i]) - 1):
#         filtered_pixel = np.sum(working_ar_l[i - 1:i + 2, j - 1:j + 2] * ph_filter)
#         filtered_ar[i, j] = np.uint8(np.clip(filtered_pixel, 0, 255))


# print(working_ar_l)
# print(filtered_ar)

filtered_ar = np.clip(filtered_ar, 0, 255).astype(np.uint8)

photo = Image.fromarray(filtered_ar)

try:
    os.mkdir('Results\\')
except FileExistsError:
    pass

photo.save('Results\\test_result.jpg')