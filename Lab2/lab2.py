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


def do_multiplications(working_ar, ph_filter):
    filtered_ar = np.zeros_like(working_ar)
    # print(filtered_ar.dtype)
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
                
                match channel:
                    case Codes.red:
                        print("Red layer started")
                    case Codes.green:
                        print("Green layer started")
                    case Codes.blue:
                        print("Blue layer started")
                        
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
                        npsum = np.sum(matrix * ph_filter)
                        filtered_ar[row, col, channel] = npsum
                
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
                        matrix = working_ar[row - height_limit:row + 2, col - width_limit:col + 2]
                        filtered_ar[row, col] = np.sum(matrix * ph_filter)
    return filtered_ar


def scale(filtered_ar1):
    print("Scaling array...")
    maximum = np.max(filtered_ar1)
    minimum = np.min(filtered_ar1)
    filtered_ar1 = (255*(filtered_ar1 - minimum)/np.ptp(filtered_ar1)).astype(np.uint8)
    return filtered_ar1
    
    
def clip(filtered_ar1):
    print("Cliping array...")
    filtered_ar1 = np.clip(filtered_ar1, 0, 255).astype(np.uint8)
    return filtered_ar1
    
################################################
photo_or = Image.open("Photos\\test1.jpg")
ph_filter = Inversia()
################################################
print()
photo = np.array(photo_or, np.float64)
try:
    os.mkdir('Results\\')
except FileExistsError:
    pass
################################################
filtered_ar1 = do_multiplications(photo, ph_filter)
# print(np.any(filtered_ar1 < -255), np.any(filtered_ar1 > 255))
if np.sum(ph_filter) == 0 or np.all(ph_filter  == Rizkist()):
    filtered_ar1 = clip(filtered_ar1)
else:
    filtered_ar1 = scale(filtered_ar1)
# print(np.any(filtered_ar1 < -255), np.any(filtered_ar1 > 255))
photo = Image.fromarray(filtered_ar1.astype(np.uint8))
################################################
photo.show()
photo.save('Results\\test_result_#' + str(dtime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")) + '.jpg')
print('\nDone!')