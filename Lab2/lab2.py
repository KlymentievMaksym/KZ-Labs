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
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32)

working_ar = np.array(photo)
working_ar_l = rgb_to_halftone(working_ar).astype(np.uint8)
print(working_ar.shape)
print(working_ar_l.shape)

filtered_ar = np.zeros_like(working_ar_l)

ar_size = len(working_ar_l)


# for i in range (0, 3):
#     for j in range(0, 3):
#         print(i, j, ph_filter[i][j])
#         # print(ph_filter[i][j])

for row in range(1, ar_size-1):
    for col in range(1, ar_size):
        if row >= ar_size - 4:
            matrix = working_ar_l[row - 1:row + 2, col - 1:col + 2]
            print(matrix, ph_filter, matrix * ph_filter, np.sum(matrix * ph_filter))
            filtered_ar[row][col] = np.sum(matrix * ph_filter)

    #         sum_by_filter = 0
#         for i in range (0, 3):
#             for j in range(0, 3):
#                 sum_by_filter += working_ar_l[(row + i - 1) % ar_size][(col + j - 1) % ar_size] * ph_filter[i][j]
#         filtered_ar[row][col] = sum_by_filter

# for i in range(1, len(working_ar_l) - 1):
#     for j in range(1, len(working_ar_l[i]) - 1):
#         filtered_pixel = np.sum(working_ar_l[i - 1:i + 2, j - 1:j + 2] * ph_filter)
#         filtered_ar[i, j] = np.uint8(np.clip(filtered_pixel, 0, 255))


print(working_ar_l)
print(filtered_ar)

filtered_ar = np.clip(filtered_ar, 0, 255).astype(np.uint8)

# photo = Image.fromarray(filtered_ar)

# try:
#     os.mkdir('Results\\')
# except FileExistsError:
#     pass

# photo.save('Results\\test_result.jpg')