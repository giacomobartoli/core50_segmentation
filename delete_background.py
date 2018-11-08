from matplotlib import pyplot as plt
from skimage import io
import numpy
import PIL
import math

name='C_05_17_022'
image_path = 'depth_images/C_05_17_022.png'
depth_image_path = 'depth_images/C_05_17_022.png'
intermedio = 'intermedio.jpg'


# SET THESE VALUES BEFORE STARTING
background_depth = 200
svm_threshold = -2.2

depth_image = numpy.asarray(PIL.Image.open(depth_image_path).convert('LA'))
depth_image.setflags(write=1)

rgb_image = PIL.Image.open(image_path)

hand_segmentation = True

# questo ciclo invece lavora solo sulla prima colonna dell'array 2D
# print(new_image.flags)
for i in range(len(depth_image)):
    for j in range(len(depth_image[i])):
        #dove la depth è < 226, la azzeriamo.
        if(depth_image[i][j][0]) < background_depth:
            depth_image[i][j][0] = 0
        #questa parte colora di nero i pixel dove manca il canale alpha
        if (depth_image[i][j][1]) == 0:
            depth_image[i][j][1] = 255
            #new_image[i][j][0] = 255


img = PIL.Image.fromarray(depth_image)
img = img.convert('RGB')
img.save("intermedio.png")


# Coloro i pixel del background della immagine RGB (128x128)
pixels=rgb_image.load()
width, height = img.size
for x in range(width):
    for y in range(height):
        r,g,b = img.getpixel((x, y))
        if r == 0 & g == 0 & b == 0:
            pixels[x, y] = (0, 0, 0)
        if r is None or g is None or b is None:
            print('pixel nullo')
            pixels[x, y] = (0, 0, 0)

rgb_image.save('risultato.png')

# applico SVM preaddestrato

if hand_segmentation:

    import pickle as cPickle

    with open('trained_SVM.pkl', 'rb') as fid:
        svm = cPickle.load(fid)
        print('SVM loaded correctly')

    list = []
    for x in range(width):
        for y in range(height):
            print(str(x) + ' ' + str(y))
            rgb_list = []
            r, g, b = rgb_image.getpixel((x, y))
            rgb_list.append(r)
            rgb_list.append(g)
            rgb_list.append(b)
            list.append(rgb_list)

    print('starting predictions..')
    #pred=svm.predict(list)
    pred = svm.decision_function(list)
    print('MIN VALUE')
    print(pred.min())
    i = 0
    pixels = rgb_image.load()
    for x in range(width):
        for y in range(height):
            if pred[i] >= svm_threshold:
                print('è la mano')
                pixels[x, y] = (0, 0, 0)
            else:
                print('non è la mano')
            i = i + 1

    # i = 0
    # pixels = rgb_image.load()
    # for x in range(width):
    #     for y in range(height):
    #         if pred[i] == 1:
    #             print('è la mano')
    #             pixels[x, y] = (0, 0, 0)
    #         else:
    #             print('non è la mano')
    #         i = i + 1

    rgb_image.save('results/FINE_'+name+'.png')

    # Convertire immagine in binaria
    # Applicare operatori morfologici

    print('done')




