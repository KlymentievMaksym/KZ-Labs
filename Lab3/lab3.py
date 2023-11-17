import numpy as np
from PIL import Image, ImageShow
import os
import datetime as dtime

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


def fit(ar3x3):
    # print(ar3x3, fl, np.all(ar3x3 == fl))
    a = np.equal(1, fl)
    b = np.equal(255, ar3x3)
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
    if np.all(result): return True
    return False


def hit(ar3x3):
    a = np.equal(1, fl)
    b = np.equal(255, ar3x3)
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
    result = result
    # result = np.logical_or(a, b)
    if np.all(result): return True
    return False


def do_morph(photo, fl, fl_type, wild_point_loc):
    
    ar_to_return = photo.copy()
    #ar_to_return[...] = 255 
    
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
                    if fit(photo[row_start:row_finish, col - width_limit:col + 2, channel]): ar_to_return[row + wild_point_loc[0]-1, col + wild_point_loc[1]-1, channel] = 255
                    else: ar_to_return[row + wild_point_loc[0]-1, col + wild_point_loc[1]-1, channel] = 0
                        
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


# ImageShow.WindowsViewer.format = "webp"


fl = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    ])

wild_point_loc = (1, 1)

# print(fl[wild_point_loc])

fl_types = np.array(['fit', 'hit']) # ['fit', 'hit']
################################################
photo = np.array(photo_or)

for fl_type in fl_types:
    photo = do_morph(photo, fl, fl_type, wild_point_loc)
    
photo = Image.fromarray(photo)
################################################
try:
    os.mkdir('Results\\')
except FileExistsError:
    pass

photo.show()
photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.webp')
print('\nDone!')