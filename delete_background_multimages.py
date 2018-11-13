import numpy
import PIL.Image
import PIL
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Creating a list of all the images to process
images_to_process = []
images_to_process = os.listdir("images")

# SET THESE VALUES BEFORE STARTING
background_depth = 200
svm_threshold = -2.2
dilated = 1


for image in sorted(images_to_process):
    print('Start processing '+image)
    image_path = 'images/' + image
    depth_name = ''
    depth_name = list(image)
    depth_name[0] = 'D'
    depth_name = ''.join(depth_name)
    depth_image_path = 'depth_images/' + str(depth_name)

    depth_image = numpy.asarray(PIL.Image.open(depth_image_path).convert('LA'))
    depth_image.setflags(write=1)
    rgb_image = PIL.Image.open(image_path)
    hand_segmentation = True

    # qui devo crearmi una immagine di 1
    binary_image = numpy.zeros([128, 128], dtype=numpy.uint8)
    binary_image.fill(1)

    # deleting pixels belonging to the background
    for i in range(len(depth_image)):
        for j in range(len(depth_image[i])):
            if (depth_image[i][j][0]) < background_depth:
                depth_image[i][j][0] = 0
            if (depth_image[i][j][1]) == 0:
                depth_image[i][j][1] = 255

    img = PIL.Image.fromarray(depth_image)
    img = img.convert('RGB')



    # Taking the respective RGB image and deleting the background
    pixels = rgb_image.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            if r == 0 & g == 0 & b == 0:
                pixels[x, y] = (0, 0, 0)
                #new_image[x,y] = 0
            else:
                binary_image[x, y] = 0

    # rgb_image.save('result.png')



    # Using pre-trained SVM model for detecting pixels belonging to the hand
    if hand_segmentation:

        import pickle as cPickle

        with open('trained_SVM.pkl', 'rb') as fid:
            svm = cPickle.load(fid)
            print('SVM loaded correctly')

        list_SVM = []
        for x in range(width):
            for y in range(height):
                # print(str(x) + ' ' + str(y))
                rgb_list = []
                r, g, b = rgb_image.getpixel((x, y))
                rgb_list.append(r)
                rgb_list.append(g)
                rgb_list.append(b)
                list_SVM.append(rgb_list)

        print('starting SVM prediction for '+image)
        pred = svm.decision_function(list_SVM)


        i = 0
        pixels = rgb_image.load()
        for x in range(width):
            for y in range(height):
                if pred[i] >= svm_threshold:
                    #print('this pixel belongs to the hand')
                    pixels[x, y] = (0, 0, 0)
                    binary_image[x, y] = 1
                i = i + 1

        rgb_image.save('results/FINE_' + image)

        from skimage import morphology

        # l'immagine risultante la passo direttamente alla morfologia
        plt.imsave('predilation/' + image, binary_image, cmap=cm.gray)


        # Applying morphological operators
        dilated_image = morphology.binary_dilation(binary_image, morphology.diamond(dilated)).astype(numpy.uint8)

        # saving the final image
        plt.imsave('dilations/' + image,dilated_image, cmap=cm.gray)

print('done')









