import numpy as np
from PIL import Image
import os
from numba import njit


@njit()
def rgb2hsv(image_rgb):
    '''
    converts rgb to hsv
    
    image_rgb: Numpy 2D array 
    returns: Numpy 2D array
    '''
    image_hsv = np.zeros_like(image_rgb)
    
    height, width, channels = image_rgb.shape
    
    # for channel in range(channels):
    for row in range(height):
        for col in range(width):
            r, g, b = image_rgb[row,col,0], image_rgb[row,col,1], image_rgb[row,col,2]
            _r, _g, _b = r/255, g/255, b/255
            c_max = max(_r, _g, _b)
            c_min = min(_r, _g, _b)
            delta = c_max - c_min
            
            # hue 
            if delta == 0:
                h = np.degrees(0)
            elif c_max == _r:
                h = np.degrees(60) * (((_g - _b)/delta)%6)
            elif c_max == _g:
                h = np.degrees(60) * (((_b - _r)+2))
            elif c_max == _b:
                h = np.degrees(60) * (((_r - _g)+4))

                # saturation 
            if c_max == 0:
                s = 0
            elif c_max != 0:
                s = delta/c_max

            # value
            v = c_max
            
            image_hsv[row,col,0], image_hsv[row,col,1], image_hsv[row,col,2] = h, s, v
    return image_hsv

@njit()
def hsv2rgb(image_hsv):
    '''
    converts hsv to rgb
    
    image_rgb: Numpy 2D array 
    returns: Numpy 2D array
    '''
    
    image_rgb = np.zeros_like(image_hsv)
    
    height, width, channels = image_hsv.shape
    
    # for channel in range(channels):
    for row in range(height):
        for col in range(width):
            h, s, v = image_hsv[row,col,0], image_hsv[row,col,1], image_hsv[row,col,2]
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
            image_rgb[row,col,0], image_rgb[row,col,1], image_rgb[row,col,2] = r, g, b
    return image_rgb


# @njit()
def rgb_to_hsv(rgb_image):
    # Normalize RGB values to the range [0, 1]
    normalized_rgb = rgb_image / 255.0

    # Extract individual color channels
    r, g, b = normalized_rgb[:, :, 0], normalized_rgb[:, :, 1], normalized_rgb[:, :, 2]

    # Compute value (brightness)
    v = np.max(normalized_rgb, axis=2)

    # Compute saturation
    s = np.where(v != 0, (v - np.min(normalized_rgb, axis=2)) / v, 0)

    # Compute hue
    delta = np.clip(v - np.min(normalized_rgb, axis=2), a_min=1e-10, a_max=None)
    delta_r, delta_g, delta_b = (v - r) / delta, (v - g) / delta, (v - b) / delta

    h = np.select(
        [r == v, g == v, b == v],
        [delta_b - delta_g, 2.0 + delta_r - delta_b, 4.0 + delta_g - delta_r],
        default=0.0,
    )

    h = (h / 6.0) % 1.0

    # Scale hue to the range [0, 179]
    h = (h * 179).astype(np.uint8)

    # Scale saturation and value to the range [0, 255]
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)

    # Stack the channels back together
    hsv_image = np.stack([h, s, v], axis=-1)

    return hsv_image



# @njit()
def hsv_to_rgb(hsv_image):
    # Normalize HSV values to the range [0, 1]
    normalized_hsv = hsv_image / 255.0

    # Extract individual channels
    H, S, V = normalized_hsv[:,:,0], normalized_hsv[:,:,1], normalized_hsv[:,:,2]

    C = V * S
    X = C * (1 - np.abs((H / 60) % 2 - 1))
    m = V - C

    # Initialize RGB values to zero
    R, G, B = 0, 0, 0

    # Define conditions for each sector in the color wheel
    conditions = [
        (0 <= H) & (H < 60),
        (60 <= H) & (H < 120),
        (120 <= H) & (H < 180),
        (180 <= H) & (H < 240),
        (240 <= H) & (H < 300),
        (300 <= H) & (H < 360)
    ]

    # Calculate RGB values based on the conditions
    R = np.select(conditions, [C, X, 0, 0, X, C])
    G = np.select(conditions, [X, C, C, X, 0, 0])
    B = np.select(conditions, [0, 0, X, C, C, X])

    # Add m to each channel
    R = (R + m) * 255
    G = (G + m) * 255
    B = (B + m) * 255

    # Stack the channels back together
    rgb_image = np.stack([R, G, B], axis=-1)

    # Convert to integer type
    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image

# def rgb2hsv(image_rgb):
#     '''
#     converts rgb to hsv
    
#     image_rgb: Numpy 2D array 
#     returns: Numpy 2D array
#     '''
#     r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
#     h, s, v = np.zeros_like(r), np.zeros_like(g), np.zeros_like(b)
#     _r, _g, _b = r/255, g/255, b/255
#     c_max = (np.max([_r, _g, _b], axis=0))
#     c_min = (np.min([_r, _g, _b], axis=0))
#     delta = c_max - c_min
#     # print(delta)
#     # print(_r[0, 1])
#     # print(len(delta[1]))
#     for i in range(len(delta[1])):
#         # hue 
#         if delta[0, i] == 0:
#             h[0, i] = np.degrees(0)
#         elif c_max[0, i] == _r[0, i]:
#             h[0, i] = np.degrees(60) * (((_g[0, i] - _b[0, i])/delta[0, i])%6)
#         elif c_max[0, i] == _g[0, i]:
#             h[0, i] = np.degrees(60) * (((_b[0, i] - _r[0, i])+2))
#         elif c_max[0, i] == _b[0, i]:
#             h[0, i] = np.degrees(60) * (((_r[0, i] - _g[0, i])+4))
        
#         # saturation 
#         if c_max[0, i] == 0:
#             s[0, i] = 0
#         elif c_max[0, i] != 0:
#             s[0, i] = delta[0, i]/c_max[0, i]
        
#         # value
#         v[0, i] = c_max[0, i]
        
#     image_hsv = np.concatenate((h, s, v))
#     return image_hsv


# def hsv2rgb(image_hsv):
#     '''
#     converts hsv to rgb
    
#     image_rgb: Numpy 2D array 
#     returns: Numpy 2D array
#     '''
#     h, s, v = image_hsv[:,:,0], image_hsv[:,:,1], image_hsv[:,:,2]
#     c = v * s
#     x = c * (1 - np.abs((h/np.degrees(60))%2 - 1))
#     m = v - c
    
#     if h < np.degrees(60) or h == np.degrees(360):
#         _r, _g, _b = c, x, 0
#     elif h < np.degrees(120):
#         _r, _g, _b = x, c, 0
#     elif h < np.degrees(180):
#         _r, _g, _b = 0, c, x
#     elif h < np.degrees(240):
#         _r, _g, _b = 0, x, c
#     elif h < np.degrees(300):
#         _r, _g, _b = x, 0, c
#     elif h < np.degrees(360):
#         _r, _g, _b = c, 0, x
        
#     r, g, b = (_r + m)*255, (_g + m)*255, (_b + m)*255
#     image_rgb = np.concatenate((r, g, b), axis=2)
#     return image_rgb

# for item in os.listdir('photos\\'):
#     if '1' in item:
#         image = np.array(Image.open('photos\\'+item))
        
#         img = Image.fromarray(rgb2hsv(image), mode='HSV')
#         img.save('results\\'+item+'_hsv_'+item)