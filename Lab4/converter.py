import numpy as np
from PIL import Image
import os

def rgb2hsv(image_rgb):
    '''
    converts rgb to hsv
    
    image_rgb: Numpy 2D array 
    returns: Numpy 2D array
    '''
    r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
    h, s, v = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
    _r, _g, _b = r/255, g/255, b/255
    c_max = (np.max([_r, _g, _b], axis=0))
    c_min = (np.min([_r, _g, _b], axis=0))
    delta = c_max - c_min
    # print(delta)
    # print(_r[0, 1])
    # print(len(delta[1]))
    for i in range(len(delta[1])):
        # hue 
        if delta[0, i] == 0:
            h[0, i] = np.degrees(0)
        elif c_max[0, i] == _r[0, i]:
            h[0, i] = np.degrees(60) * (((_g[0, i] - _b[0, i])/delta[0, i])%6)
        elif c_max[0, i] == _g[0, i]:
            h[0, i] = np.degrees(60) * (((_b[0, i] - _r[0, i])+2))
        elif c_max[0, i] == _b[0, i]:
            h[0, i] = np.degrees(60) * (((_r[0, i] - _g[0, i])+4))
        
        # saturation 
        if c_max[0, i] == 0:
            s[0, i] = 0
        elif c_max[0, i] != 0:
            s[0, i] = delta[0, i]/c_max[0, i]
        
        # value
        v[0, i] = c_max[0, i]
        
    image_hsv = np.concatenate((h, s, v))
    return image_hsv


def hsv2rgb(image_hsv):
    '''
    converts hsv to rgb
    
    image_rgb: Numpy 2D array 
    returns: Numpy 2D array
    '''
    h, s, v = image_hsv[:,:,0], image_hsv[:,:,1], image_hsv[:,:,2]
    c = v * s
    x = c * (1 - np.abs((h/np.degrees(60))%2 - 1))
    m = v - c
    
    if h < np.degrees(60) or h == np.degrees(360):
        _r, _g, _b = c, x, 0
    elif h < np.degrees(120):
        _r, _g, _b = x, c, 0
    elif h < np.degrees(180):
        _r, _g, _b = 0, c, x
    elif h < np.degrees(240):
        _r, _g, _b = 0, x, c
    elif h < np.degrees(300):
        _r, _g, _b = x, 0, c
    elif h < np.degrees(360):
        _r, _g, _b = c, 0, x
        
    r, g, b = (_r + m)*255, (_g + m)*255, (_b + m)*255
    image_rgb = np.concatenate((r, g, b), axis=2)
    return image_rgb

for item in os.listdir('photos\\'):
    if '1' in item:
        image = np.array(Image.open('photos\\'+item))
        
        img = Image.fromarray(rgb2hsv(image), mode='HSV')
        img.save('results\\'+item+'_hsv_'+item)