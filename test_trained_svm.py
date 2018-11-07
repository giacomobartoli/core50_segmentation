import PIL.Image
import pickle as cPickle
import numpy as np


image_path = 'C_10_50_294.jpg'
list = []
newimdata = []

with open('trained_SVM.pkl', 'rb') as fid:
    svm = cPickle.load(fid)
    print('SVM loaded correctly')

im = PIL.Image.open(image_path)
width, height = im.size
for x in range(width):
    for y in range(height):
        print(str(x)+' '+str(y))
        rgb_list = []
        r, g, b = im.getpixel((x, y))
        rgb_list.append(r)
        rgb_list.append(g)
        rgb_list.append(b)
        list.append(rgb_list)

print('starting predictions..')
pred=svm.predict(list)

i=0
pixels=im.load()
for x in range(width):
    for y in range(height):
        if pred[i] == 1:
            print('è la mano')
            pixels[x,y] = (255, 255, 255)
        else:
            print('non è la mano')
        i=i+1

im.save('test.png')

print('done')
