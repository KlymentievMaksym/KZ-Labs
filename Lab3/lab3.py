import numpy as np
from PIL import Image, ImageShow
import os
import datetime as dtime

from pprint import pprint
from fnmatch import fnmatch
# from tqdm import tqdm
from numba import njit
# from numba_progress import ProgressBar

@njit(nogil=True)
def eq(ar1, ar2):
    a1, a2, a3 = ar1
    b1, b2, b3 = ar2
    
    lim1 = (a1 and b1 or not b1)
    lim2 = (a2 and b2 or not b2)
    lim3 = (a3 and b3 or not b3)
    
    # print(a1, a2, a3, lim1, lim2, lim3)
    # print(b1, b2, b3)
    
    if not lim1 or not lim2 or not lim3: return False
    return True

@njit(nogil=True)
def h_eq(ar1, ar2):
    a1, a2, a3 = ar1
    b1, b2, b3 = ar2
    
    lim1 = (a1 and b1 or not b1)
    lim2 = (a2 and b2 or not b2)
    lim3 = (a3 and b3 or not b3)
    
    # print(a1, a2, a3, lim1, lim2, lim3)
    # print(b1, b2, b3)
    
    if lim1 or lim2 or lim3: return True
    return False

@njit(nogil=True)
def fit(ar3x3):
    # print(ar3x3, fl, np.all(ar3x3 == fl))
    a = np.equal(1, fl)
    b = np.equal(0, ar3x3)
    result = []
    
    if not np.any(a[:, 0]):
        pass
    else:
        result.append(eq(b[:, 0], a[:, 0]))
    
    if not np.any(a[:, 1]):
        pass
    else:
        result.append(eq(b[:, 1], a[:, 1]))
        
    if not np.any(a[:, 2]):
        pass
    else:
        result.append(eq(b[:, 2], a[:, 2]))
        
    # print(result, np.all(result))
    
    # result = np.logical_or(a, b)
    is_true1 = False
    is_true2 = False
    is_true3 = False
    
    for boolean in result:
        if boolean: is_true1 = True
        elif is_true1 and boolean: is_true2 = True
        elif is_true1 and is_true2 and boolean: is_true3 = True
        
    
    if is_true1 and is_true2 and is_true3: return True
    return False

@njit(nogil=True)
def hit(ar3x3):
    a = np.equal(1, fl)
    b = np.equal(0, ar3x3)
    result = []
    
    if not np.any(a[:, 0]):
        pass
    else:
        result.append(h_eq(b[:, 0], a[:, 0]))
    
    if not np.any(a[:, 1]):
        pass
    else:
        result.append(h_eq(b[:, 1], a[:, 1]))
        
    if not np.any(a[:, 2]):
        pass
    else:
        result.append(h_eq(b[:, 2], a[:, 2]))
        
    # print(result, np.all(result))
    # result = result
    # result = np.logical_or(a, b)
    is_true = False
    
    for boolean in result:
        if boolean: is_true = True
    
    if is_true: return True
    return False

@njit(nogil=True)
def do_morph(photo, fl, fl_type, wild_point_loc, not_rgb=False):
    ar_to_return = photo.copy()
    #ar_to_return[...] = 255 
    
    tupl = ar_to_return.shape
    # print(len(tupl))
    
    # if len(tupl) == 3:
    height, width, channels = photo.shape
    # else:
        # height, width = photo.shape
        # not_rgb = True
        # channels = 1
        
    height_fl, width_fl = fl.shape
    
    # print(height, width, channels)
    
    height_limit = height_fl - 2
    width_limit = width_fl - 2
    
    # print(height_fl, width_fl, height_limit, width_limit)
    
    # print(fl)

    
    if fl_type == 'fit':
        print('FIT start')
        for channel in range(channels):
            for row in range(height_limit, height - height_limit):
                
                row_start = row - height_limit
                row_finish = row + 2
                
                for col in range(width_limit, width - width_limit):
                    if not_rgb:
                        if fit(photo[row_start:row_finish, col - width_limit:col + 2]): ar_to_return[row + wild_point_loc[0]-1, col + wild_point_loc[1]-1] = 0
                        else: ar_to_return[row + wild_point_loc[0]-1, col + wild_point_loc[1]-1] = 255
                    else:
                        if fit(photo[row_start:row_finish, col - width_limit:col + 2, channel]): ar_to_return[row + wild_point_loc[0]-1, col + wild_point_loc[1]-1, channel] = 0
                        else: ar_to_return[row + wild_point_loc[0]-1, col + wild_point_loc[1]-1, channel] = 255
                        
    elif fl_type == 'hit':
        print('HIT start')
        for channel in range(channels):
            for row in range(height_limit, height - height_limit):
                
                row_start = row - height_limit
                row_finish = row + 2
                
                for col in range(width_limit, width - width_limit):
                    if not_rgb:
                        if hit(photo[row_start:row_finish, col - width_limit:col + 2]): ar_to_return[row, col] = 0
                        else: ar_to_return[row, col] = 255
                    else:
                        if hit(photo[row_start:row_finish, col - width_limit:col + 2, channel]): ar_to_return[row, col, channel] = 0
                        else: ar_to_return[row, col, channel] = 255
                    
    
    return ar_to_return


################################################
photo_or = Image.open("Photos\\MyART1.jpg")


ImageShow.WindowsViewer.format = "webp"


fl = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    ])

wild_point_loc = (0, 1)

fl_types = np.array(['fit', 'it']) # ['fit', 'hit']
################################################
photo = np.array(photo_or)

# photo = np.zeros((20, 20))
# photo[:, :] = 255
# photo[3:6, 2:5] = 0
# photo[5:10, 4:8] = 0

# photo = rgb_to_halftone(photo)

for fl_type in fl_types:
    if fl_type == 'hitmfit':
        hits = do_morph(photo, fl, 'hit', wild_point_loc)
        fits = do_morph(photo, fl, 'fit', wild_point_loc)
        photo = hits-fits # ).astype(np.uint8)
        # photo[photo==0] = 255
        photo[photo==1] = 255
    else:
        photo = do_morph(photo, fl, fl_type, wild_point_loc)


# print(photo)

photo = Image.fromarray(photo)
################################################
try:
    os.mkdir('Results\\')
except FileExistsError:
    pass

photo.show()
photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + fl_types[0] + fl_types[1] +'.webp')
print('\nDone!')