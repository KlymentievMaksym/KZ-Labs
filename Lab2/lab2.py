import numpy as np
from PIL import Image

from kernels import *

from pprint import pprint
from fnmatch import fnmatch
# from tqdm import tqdm
from numba import njit
from numba_progress import ProgressBar

import os
import datetime as dtime


def rgb_to_halftone(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    halftone = 0.36 * r + 0.53 * g + 0.11 * b
    # halftone = (r + g + b)/3

    return halftone

@njit(nogil=True)
def apply_filter(working_ar, ph_filter, filtered_ar, height, width, channels, height_limit, width_limit, progress):
    for channel in range(channels):
        for row in range(height_limit, height - height_limit):
            
            row_start = row - height_limit
            row_finish = row + 2
            
            for col in range(width_limit, width - width_limit):
                matrix = working_ar[row_start:row_finish, col - width_limit:col + 2, channel]
                npsum = np.sum(matrix * ph_filter)
                filtered_ar[row, col, channel] = npsum
                progress.update(1)
    return filtered_ar


# @njit(nogil=True)
def do_multiplications(working_ar, ph_filter):
    
    filtered_ar = np.zeros_like(working_ar)
    height, width, channels = working_ar.shape
    
    height_fl, width_fl = ph_filter.shape
    
    height_limit = height_fl - 2
    width_limit = width_fl - 2
    
    with ProgressBar(total=(height - 2*height_limit)*(width - 2*width_limit)*channels) as progress:
        return apply_filter(working_ar, ph_filter, filtered_ar, height, width, channels, height_limit, width_limit, progress)

@njit(nogil=True)
def scale(filtered_ar1):
    # maximum = np.max(filtered_ar1)
    minimum = np.min(filtered_ar1)
    filtered_ar1 = (255*(filtered_ar1 - minimum)/np.ptp(filtered_ar1)).astype(np.uint8)
    return filtered_ar1

    
@njit(nogil=True)  
def clip(filtered_ar1):
    filtered_ar1 = np.clip(filtered_ar1, 0, 255).astype(np.uint8)
    return filtered_ar1


# @njit(nogil=True)
def find_maybe_names(res_listdir, future_name):
    examples = []
    for filename in res_listdir:
        if fnmatch(filename, future_name+'.*'):
            examples += [filename]
    return examples


def filenamecheckget(res_listdir, to_continue):
    filename = input('File Name: ')
    maybe_names = find_maybe_names(res_listdir, filename)
    if maybe_names == []:
        return filenamecheckget(res_listdir, to_continue)
    if filename == "":
        return False
    return filename, maybe_names


################################################
kernels_dict = {
    1:ZSUv(),
    2:Inversia(),
    3:Gauss_11x11(),
    4:Move_diagonal(),
    5:Rizkist(),
    6:Sobel_diag(),
    7:Border(),
    8:My_Kernel()
    }
kernels_dict_name = {
    1:'ZSUv',
    2:'Inversia',
    3:'Gauss 11x11',
    4:'Moving diagonal',
    5:'Sharpening',
    6:'Sobel diagonal',
    7:'Border',
    8:'My Kernel'
    }
################################################
try:
    os.mkdir('Results\\')
except FileExistsError:
    pass

to_continue = True

while to_continue:
    
    res_listdir = os.listdir('Photos\\')
    
    filename, maybe_names = filenamecheckget(res_listdir, to_continue)
    if filename is False:
        to_continue = False
        continue
    
    print()
    pprint(maybe_names)
    print()
    
    filetype = input('File Type: ')
    if filetype == "":
        to_continue = False
        continue
    
    print()
    print('Codes for kernels: ')
    pprint(kernels_dict_name)
    print()
    
    kerneltype = input('Kernel: ')
    if kerneltype == "":
        to_continue = False
        continue
    
    print()
    print(kernels_dict_name[int(kerneltype)])
    
    photo_or = Image.open("Photos\\"+filename+"."+filetype)
    ph_filter = kernels_dict[int(kerneltype)]
    ################################################
    photo = np.array(photo_or, np.float64)
    filtered_ar1 = do_multiplications(photo, ph_filter)
    print()
    if np.sum(ph_filter) == 0 or (ph_filter.shape == Rizkist().shape and np.all(ph_filter  == Rizkist())):
        print("Cliping array...")
        filtered_ar1 = clip(filtered_ar1)
    else:
        print("Scaling array...")
        filtered_ar1 = scale(filtered_ar1)
    photo = Image.fromarray(filtered_ar1.astype(np.uint8))
    ################################################
    photo.show()
    photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.jpg')
    print('Done!\n')