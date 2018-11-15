import csv
import os
import xml.etree.ElementTree
from fnmatch import fnmatch
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import PIL.Image
import pickle as cPickle
import numpy as np

# Header for the .csv file:
# Choose your header by comment/uncomment lines below.
column_name = ['Filename', 'class', 'points']
column_name2 = ['']

# lists for PR
y_scores = []
y_true = []

#list of polygons
polygons=[]

root = 'training_images'
pattern = "*.jpg"

# Bounding boxes root
bbox = ''
pattern_bbox = "*.xml"

# This is an empty list that will be filled with all the data: filename, width, height, session..etc
filenames = []
filenames2 = []

#pixels that belong to the hand
hand_list=[]

#pixels that DO NOT belong to the hand
background_list=[]


create_dataset=False
create_dataset2=False
plot_pixels = False
plot_pixels2 = False
train_SVM = True
# Qui decido se caricare gli array da FS oppure ricrearli da zero
create_pixels = False
precision_recall = True

# extract random elements from list of outliers
import random

def random_elements(list):
    list_rand = []
    for i in range(10000):
        list_rand.append(random.choice(list))
    return list_rand

# given an image and a polygon it checks if pixels belong to that segment
def check_polygon_for_image(image_path,polygon):
    im = PIL.Image.open(image_path)
    width, height = im.size
    for x in range(width):
        for y in range(height):
            #print('Checking pixel '+str(x)+' '+str(y))
            point = Point(x, y)
            poly = MultiPoint(polygon).convex_hull
            #print(poly.contains(point))
            if poly.contains(point):
                rgb_list = []
                r, g, b = im.getpixel((x, y))
                rgb_list.append(r)
                rgb_list.append(g)
                rgb_list.append(b)
                hand_list.append(rgb_list)
                print(str(r)+ ' '+ str(g)+ ' '+str(b))
            else:
                rgb_list = []
                r, g, b = im.getpixel((x, y))
                rgb_list.append(r)
                rgb_list.append(g)
                rgb_list.append(b)
                background_list.append(rgb_list)
    np.save('handlist.npy', hand_list)
    np.save('background_list.npy',background_list)



# scanning the file system, creating a list with all polygons
for path, subdirs, files in os.walk(root):
    #print(files)
    for name in sorted(files):
         if fnmatch(name, pattern):
             #print(name)
             listToAppend = []
             temp_list =[]
             listToAppend.append(name)
             listToAppend.append('hand')
             xml_name=root+'/'+name[:-4]+'.xml'
             #print(xml_name)
             e = xml.etree.ElementTree.parse(xml_name).getroot()
             for i in range(24):
                 x_value='x'+str(i+1)
                 y_value='y'+str(i+1)
                 for h in e.iter(x_value):
                     #print(h.text)
                     temp_list_coordinates = []
                     temp_list_coordinates.append(float(h.text))
                 for h in e.iter(y_value):
                     #print(h.text)
                     temp_list_coordinates.append(float(h.text))
                 temp_list.append(temp_list_coordinates)
             polygons.append(temp_list)
             listToAppend.append(temp_list)
             filenames.append(listToAppend)




# Checking all the polygons
for idx, val in enumerate(polygons):
    print("POLIGONO "+str(idx))
    poly = MultiPoint(polygons[idx]).convex_hull



for path, subdirs, files in os.walk(root):
    #print(files)
    i = 0
    for name in sorted(files):
         if fnmatch(name, pattern):
             #new_list.append(name)
             if create_pixels:
                check_polygon_for_image(name, polygons[i])
             else:
                 # load data from disk..
                 print('loading arrays from disk..')
                 hand_list=np.load('handlist.npy')
                 background_list=np.load('background_list.npy')
             print(i)
             i+=1




import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

if train_SVM:
    print("Start training one_class SVM..")

    xx, yy = np.meshgrid(np.linspace(0, 255, 255), np.linspace(0, 255, 255))
    #xx = np.meshgrid(np.linspace(0, 255, 255))
    #yy = np.meshgrid(np.linspace(0, 255, 3))

    # Splitting test and training set
    size = len(hand_list)
    train_size = int((size/100)*80)
    test_size = int((size/100)*20)
    X_train=hand_list[0:train_size]
    X_test=hand_list[train_size:]
    X_outliers = random_elements(background_list)
    #np.save('outliers.npy', X_outliers)
    #np.load('testnp.npy')
    print('OUTLIERS')
    print(len(X_outliers))
    #print(X_train.shape[1]==X_test.shape[1])

    print('TOTAL SIZE: ' + str(len(hand_list) + len(background_list)))
    print('PIXELS BELONGING TO THE HAND: ' + str(len(hand_list)))
    print('TRAIN_SET SIZE: ' + str(train_size))
    print('TEST_SET SIZE: ' + str(test_size))

    # fit the model
    clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma='auto')
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    # Precision/Recall arrays (test + outliers)
    y_scores_test = clf.decision_function(X_test)
    y_scores_outliers = clf.decision_function(X_outliers)
    y_scores = np.concatenate([y_scores_test, y_scores_outliers])

    #y_true_test = y_pred_test
    #y_true_outliers = y_pred_outliers
    y_true_test = np.ones(len(y_scores_test))
    y_true_outliers = np.empty(len(y_scores_outliers))
    y_true_outliers.fill(-1)
    y_true = np.concatenate([y_true_test,y_true_outliers])


    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    perc_n_error_train = (n_error_train / train_size)*100
    perc_n_error_test = (n_error_test/test_size)*100
    perc_n_error_outliers = (n_error_outliers/len(X_outliers))*100

    print('n_error_train: ' + str(n_error_train))
    print('n_error_test: ' + str(n_error_test))
    print('n_error_outliers: ' + str(n_error_outliers))

    print('% error train: '+str(perc_n_error_train)+' %')
    print('% error test: '+str(perc_n_error_test)+' %')
    print('% error outliers: '+str(perc_n_error_outliers)+' %')

    print('Saving the model on disk...')
    # save the classifier
    with open('trained_SVM.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)

    # load it again
    #with open('trained_SVM.pkl', 'rb') as fid:
    #    clf_loaded = cPickle.load(fid)

    print('Model saved on the disk!')

    #result = loaded_model.score(X_test, Y_test)
    #print(result)

    if precision_recall:
        import numpy as np
        from sklearn.metrics import precision_recall_curve
        import matplotlib.pyplot as plt
        from sklearn.utils.fixes import signature
        from sklearn.metrics import average_precision_score


        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        plt.show()


    if plot_pixels2:
        # plot the line, the points, and the nearest vectors to the plane
        # QUI SALTA!!!
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.title("Novelty Detection")
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
        plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

        s = 40
        #b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
        #b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
        #                 edgecolors='k')
        #c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
        #                  edgecolors='k')
        plt.axis('tight')
        plt.xlim((0, 255))
        plt.ylim((0, 255))
        #plt.legend([a.collections[0], b1, b2, c],
        #           ["learned frontier", "training observations",
        #            "new regular observations", "new abnormal observations"],
        #           loc="upper left",
        #           prop=matplotlib.font_manager.FontProperties(size=11))
        plt.xlabel(
            "error train: %d ; errors test: %d ; "
            "error outliers: %d"
            % (n_error_train, n_error_test, n_error_outliers))
        plt.show()
