import numpy as np
from PIL import Image
import os
import datetime as dtime


def fit(ar3x3):
    if np.all(ar3x3 == fl): return True
    return False


def hit(ar3x3):
    if np.any(ar3x3 == fl): return True
    return False


def do_morph(photo, fl, fl_type):
    
    ar_to_return = np.zeros_like(photo)
    
    height, width, channels = photo.shape
    height_fl, width_fl = fl.shape
    
    print(height, width, channels)
    
    height_limit = height_fl - 2
    width_limit = width_fl - 2
    
    print(height_fl, width_fl, height_limit, width_limit)
    
    print(fl)

    
    if fl_type == 'fit':
        print('FIT start')
        for channel in range(channels):
            for row in range(height_limit, height - height_limit):
                
                row_start = row - height_limit
                row_finish = row + 2
                
                for col in range(width_limit, width - width_limit):
                    if fit(photo[row_start:row_finish, col - width_limit:col + 2, channel]): ar_to_return[row, col, channel] = 255
                    else: ar_to_return[row, col, channel] = 0
                        
    elif fl_type == 'hit':
        print('HIT start')
        for channel in range(channels):
            for row in range(height_limit, height - height_limit):
                
                row_start = row - height_limit
                row_finish = row + 2
                
                for col in range(width_limit, width - width_limit):
                    if hit(photo[row_start:row_finish, col - width_limit:col + 2, channel]): ar_to_return[row, col, channel] = 255
                    else: ar_to_return[row, col, channel] = 0
    
    return ar_to_return


################################################
photo_or = Image.open("Photos\\MyART.webp")

fl = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    ])

fl_type = 'hit' # 'fit', 'hit'
################################################
photo = np.array(photo_or)

tmp_ar = do_morph(photo, fl, fl_type)

photo = Image.fromarray(tmp_ar)
################################################
try:
    os.mkdir('Results\\')
except FileExistsError:
    pass

photo.show()
photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.webp')
print('\nDone!')