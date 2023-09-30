import numpy as np
from PIL import Image

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

black_background = Image.new("L", (960, 1280), 0)

txt1 = Image.open("Photos\Text1.jpg")
txt2 = Image.open("Photos\Text2.jpg")
img1 = Image.open("Photos\Img1.jpg")
img2 = Image.open("Photos\Img2.jpg")

ar_txt1 = np.array(txt1)
ar_txt2 = np.array(txt2)
ar_img1 = np.array(img1)
ar_img2 = np.array(img2)

ht_txt1 = rgb_to_halftone(ar_txt1)
ht_txt2 = rgb_to_halftone(ar_txt2)
ht_img1 = rgb_to_halftone(ar_img1)
ht_img2 = rgb_to_halftone(ar_img2)

ht_txt1 = ht_txt1.astype(np.uint8)
ht_txt2 = ht_txt2.astype(np.uint8)
ht_img1 = ht_img1.astype(np.uint8)
ht_img2 = ht_img2.astype(np.uint8)


ht_img_txt1 = Image.fromarray(ht_txt1)
ht_img_txt2 = Image.fromarray(ht_txt2)
ht_img_img1 = Image.fromarray(ht_img1)
ht_img_img2 = Image.fromarray(ht_img2)

[p_i_txt1, p_i_index_txt1] = find_p_i(ht_img_txt1)
[p_i_txt2, p_i_index_txt2] = find_p_i(ht_img_txt2)
[p_i_img1, p_i_index_img1] = find_p_i(ht_img_img1)
[p_i_img2, p_i_index_img2] = find_p_i(ht_img_img2)

treshholds_txt1 =  find_treshold(p_i_txt1, p_i_index_txt1)
treshholds_txt2 =  find_treshold(p_i_txt2, p_i_index_txt2)
treshholds_img1 =  find_treshold(p_i_img1, p_i_index_img1)
treshholds_img2 =  find_treshold(p_i_img2, p_i_index_img2)

optimal_treshhold_txt1 = find_optimal_treshhold(treshholds_txt1)
optimal_treshhold_txt2 = find_optimal_treshhold(treshholds_txt2)
optimal_treshhold_img1 = find_optimal_treshhold(treshholds_img1)
optimal_treshhold_img2 = find_optimal_treshhold(treshholds_img2)

bw_img_txt1 = ht_img_txt1.point(lambda i:255 if i>optimal_treshhold_txt1 else 0)
# bw_img_txt1.show()
bw_img_txt2 = ht_img_txt2.point(lambda i:255 if i>optimal_treshhold_txt2 else 0)
# bw_img_txt2.show()
bw_img_img1 = ht_img_img1.point(lambda i:255 if i>optimal_treshhold_img1 else 0)
# bw_img_img1.show()
bw_img_img2 = ht_img_img2.point(lambda i:255 if i>optimal_treshhold_img2 else 0)
# bw_img_img2.show()

rslt_img_img1 = Image.composite(black_background, img1, bw_img_img1)
rslt_img_img2 = Image.composite(black_background, img2, bw_img_img2)
rslt_img_txt1 = Image.composite(black_background, txt1, bw_img_txt1)
rslt_img_txt2 = Image.composite(black_background, txt2, bw_img_txt2)
# rslt_img_txt1.show()

try:
    open('Photos\\ht_Text1.jpg')
except FileNotFoundError:
    ht_img_txt1.save('Photos\\ht_Text1.jpg')
try:
    open('Photos\\ht_Text2.jpg')
except FileNotFoundError:
    ht_img_txt2.save('Photos\\ht_Text2.jpg')
try:
    open('Photos\\ht_Img1.jpg')
except FileNotFoundError:
    ht_img_img1.save('Photos\\ht_Img1.jpg')
try:
    open('Photos\\ht_Img2.jpg')
except FileNotFoundError:
    ht_img_img2.save('Photos\\ht_Img2.jpg')

try:
    open('Photos\\bw_Text1.jpg')
except FileNotFoundError:
    bw_img_txt1.save('Photos\\bw_Text1.jpg')
try:
    open('Photos\\bw_Text2.jpg')
except FileNotFoundError:
    bw_img_txt2.save('Photos\\bw_Text2.jpg')
try:
    open('Photos\\bw_Img1.jpg')
except FileNotFoundError:
    bw_img_img1.save('Photos\\bw_Img1.jpg')
try:
    open('Photos\\bw_Img2.jpg')
except FileNotFoundError:
    bw_img_img2.save('Photos\\bw_Img2.jpg')
    
try:
    open('Photos\\rs_Text1.jpg')
except FileNotFoundError:
    rslt_img_txt1.save('Photos\\rs_Text1.jpg')
try:
    open('Photos\\rs_Text2.jpg')
except FileNotFoundError:
    rslt_img_txt2.save('Photos\\rs_Text2.jpg')
try:
    open('Photos\\rs_Img1.jpg')
except FileNotFoundError:
    rslt_img_img1.save('Photos\\rs_Img1.jpg')
try:
    open('Photos\\rs_Img2.jpg')
except FileNotFoundError:
    rslt_img_img2.save('Photos\\rs_Img2.jpg')