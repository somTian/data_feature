from skimage.feature import local_binary_pattern
import cv2
from skimage import feature
import numpy as np
import os
import image
import scipy
from scipy.stats import itemfreq

# settings for LBP
radius = 3
n_points = 8 * radius


def describe(image,num_points,radius , eps=1e-7):
    lbp = feature.local_binary_pattern(image, num_points, radius, method="nri_uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, num_points + 3), range=(0, num_points + 2))

    hist = hist.astype("float")
    hist = hist / (hist.sum() + eps)
    return hist


def readLBP(file):
    # desc = describe(24, 3)
    # query = cv2.imread("/home/arpit/Desktop/mini_fashion_search_engine/testing/query_03.jpg")
    query = cv2.imread(file)
    gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    hist = describe(gray, 8, 2)
    return hist



def blogLBP(image):
    im = cv2.imread(image)
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # radius = 3
    # Number of points to be considered as neighbourers
    # no_points = 8 * radius
    # Uniform LBP is used
    no_points = 8
    radius = 2
    lbp = local_binary_pattern(im_gray, no_points, radius, method='nri_uniform')  # nri_uniform' is non-rotation invariant uniform and will give 59 dimensions vector,
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # # Append image path in X_name
    # X_name.append(train_image)
    # # Append histogram to X_name
    # X_test.append(hist)
    return hist


img_path = 'data/test/'
img_list = os.listdir(img_path)
for img in img_list:
    # img_f = image.load_img(img_path+img, target_size=(224, 224))
    # lbpvec = readLBP(img_path+img)
    lbpvec = blogLBP(img_path+img)
    print(lbpvec.shape)

