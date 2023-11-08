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


def find_p_i(ht_img_img):
    p = dict()
    hst = ht_img_img.histogram()
    max_p_i = -1
    max_p_i_index = -1
    for i in range(0, 256):
        p[i] = hst[i]/(sum(hst))
        if max_p_i < p[i]:
            max_p_i = p[i]
            max_p_i_index = i
    return [p, max_p_i_index]


def find_treshold(p_i, imax):    
    
    dict_for_ob = dict()
    
    for t in range(0, imax+1):
        q1 = 0
        q2 = 0
        u1 = 0
        u2 = 0
        # o1_2 = 0
        # o2_2 = 0
        ob_2 = 0
        for i in range(0, t+1): 
            q1 += p_i[i]
        for i in range(t+1, imax+1):
            q2 += p_i[i]
        for i in range(0, t+1):
            if q1 == 0:
                u1 += ((i*p_i[i])/0.0001)
            else:
                u1 += ((i*p_i[i])/q1)
        for i in range(t+1, imax+1):
            if q2 == 0:
                u2 += ((i*p_i[i])/0.0001)
            else:
                u2 += ((i*p_i[i])/q2)
        # for _ in range(0, t+1):
        # for _ in range(t+1, imax+1):
        ob_2 = q1 * q2 * (u1-u2)**2
        dict_for_ob[t] = ob_2
    return dict_for_ob


def find_optimal_treshhold(treshholds):
    max_treshold = 0
    max_treshold_index = 0
    for i in range(0, len(treshholds)):
        if treshholds[i] > max_treshold:
            max_treshold = treshholds[i]
            max_treshold_index = i

    return max_treshold_index

def Trash():
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
              
def do_multiplications(working_ar, ph_filter):
    filtered_ar = np.zeros_like(working_ar)
    if type(ph_filter) == np.ndarray:
        
        try:
            height, width, channels = working_ar.shape
            
            height_fl, width_fl = ph_filter.shape
            
            height_limit = height_fl - 2
            width_limit = width_fl - 2
            
            class Codes:
                height_25 = height//4
                height_50 = height//2
                height_75 = (3*height)//4
                height_100 = height - height_limit - 1
                red = 0
                green = 1
                blue = 2
            
            for channel in range(channels):
                for row in range(height_limit, height - height_limit):
                    
                    match row:
                        case Codes.height_25:
                            print("25% Done")
                        case Codes.height_50:
                            print("50% Done")
                        case Codes.height_75:
                            print("75% Done")
                        case Codes.height_100:
                            print("100% Done")
                            
                    for col in range(height_limit, width - height_limit):
                        matrix = working_ar[row - height_limit:row + 2, col - width_limit:col + 2, channel]
                        filtered_ar[row, col, channel] = np.sum(matrix * ph_filter)
                
                match channel:
                    case Codes.red:
                        print("Red layer done")
                        print()
                    case Codes.green:
                        print("Green layer done")
                        print()
                    case Codes.blue:
                        print("Blue layer done")
                        print()
                        
        except ValueError:
            height, width = working_ar.shape
            channels = 1
            
            height_fl, width_fl = ph_filter.shape
            
            height_limit = height_fl - 2
            width_limit = width_fl - 2
            
            for channel in range(channels):
                for row in range(height_limit, height - height_limit):
                    
                    match row:
                        case Codes.height_25:
                            print("25% Done")
                        case Codes.height_50:
                            print("50% Done")
                        case Codes.height_75:
                            print("75% Done")
                        case Codes.height_100:
                            print("100% Done")
                    
                    for col in range(height_limit, width - height_limit):
                        matrix = working_ar[row - height_limit:row + 2, col - width_limit:col + 2]
                        filtered_ar[row, col] = np.sum(matrix * ph_filter)
    return filtered_ar
    
    
####
photo_or = Image.open("Photos\\test1.jpg")
ph_filter = Gauss_11x11()
####

photo = np.array(photo_or)

# photo_ht_txt = rgb_to_halftone(photo)
# photo_ht_txt = photo_ht_txt.astype(np.uint8)
# photo_ht = Image.fromarray(photo_ht_txt)

# [p_i_txt, p_i_index_txt] = find_p_i(photo_ht)
# treshholds_txt =  find_treshold(p_i_txt, p_i_index_txt)
# optimal_treshhold_txt = find_optimal_treshhold(treshholds_txt)

# photo_bw = photo_ht.point(lambda i:255 if i>optimal_treshhold_txt else 0)

# # photo_bw.show()

# filtered_ar = do_multiplications(photo_ht_txt, ph_filter)
# photo_ht = Image.fromarray(filtered_ar)

# photo_masked = Image.alpha_composite(photo_ht.convert('RGBA'), photo_or.convert('RGBA'))

####
# photo = Image.fromarray(filtered_ar)
# photo_or.show()
# photo_ht.show()
# photo_bw.show()
# photo_masked.show()
try:
    os.mkdir('Results\\')
except FileExistsError:
    pass
# photo_masked.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.png')
####

filtered_ar1 = do_multiplications(photo, ph_filter)
photo = Image.fromarray(filtered_ar1)
photo.show()
photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.jpg')
