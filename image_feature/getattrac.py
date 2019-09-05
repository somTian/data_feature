import os
import cv2
import numpy as np
import math
from skimage.feature import local_binary_pattern
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.stats import itemfreq
from multiprocessing import Pool



# 获取图片亮度
def get_brightness_feature(path):
    img = cv2.imread(path)
    YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    sum = 0.0
    count = 0.0
    for i in range(0, len(YUV)):
        for j in range(0, len(YUV[0])):
            element = YUV[i][j]
            count = count + 1
            sum = sum + element[0]
    return sum / count

# 获取图片饱和度
def get_saturation_feaure(path):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sum = 0.0
    count = 0.0
    for i in range(0, len(hsv)):
        for j in range(0, len(hsv[0])):
            element = hsv[i][j]
            count = count + 1
            sum = sum + element[1]
    return sum / count

# 获取图片色彩
def get_colorfulness_feature(path):
    imgBGR = cv2.imread(path)
    img = np.array(imgBGR, dtype='int64')
    rg = []
    yb = []
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            element = img[i][j]
            rg.append(float(element[2] - element[1]))
            temp = float(element[2] + element[1])
            yb.append(temp / 2 - element[0])
    numrg = np.array(rg)
    numyb = np.array(yb)
    drg = np.std(numrg)
    dyb = np.std(numyb)
    arg = np.average(numrg)
    ayb = np.average(numyb)
    cf = math.sqrt(math.pow(drg, 2) + math.pow(dyb, 2)) + 0.3 * math.sqrt(math.pow(arg, 2) + math.pow(ayb, 2))
    return cf

# 获取图片自然程度
def get_naturalness_feature(path):
    img = cv2.imread(path)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    skin = []
    grass = []
    sky = []
    count = 0.0
    for i in range(0, len(hls)):
        for j in range(0, len(hls[0])):
            element = hls[i][j]
            count = count + 1
            if element[1] >= 20 and element[1] <= 80 and element[2] > 0.1:
                if element[0] >= 25 and element[0] <= 70:
                    skin.append(element[2])
                elif element[0] >= 95 and element[0] <= 135:
                    grass.append(element[2])
                elif element[0] >= 185 and element[0] <= 260:
                    sky.append(element[2])
                else:
                    pass
    if len(skin) == 0:
        askin = 0.0
    else:
        askin = np.average(np.array(skin))

    if len(grass) == 0:
        agrass = 0.0
    else:
        agrass = np.average(np.array(grass))

    if len(sky) == 0:
        asky = 0.0
    else:
        asky = np.average(np.array(sky))

    n_skin = math.pow(math.e, -0.5 * math.pow((askin - 0.76) / 0.52, 2))
    n_grass = math.pow(math.e, -0.5 * math.pow((agrass - 0.81) / 0.53, 2))
    n_sky = math.pow(math.e, -0.5 * math.pow((asky - 0.43) / 0.22, 2))
    n = (len(skin) / count) * n_skin + (len(grass) / count) * n_grass + (len(sky) / count) * n_sky
    return n

# 获取图片对比度
def get_contrast_feature(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = np.average(np.array(gray))
    sum = 0.0
    count = 0.0
    for i in range(0, len(gray)):
        for j in range(0, len(gray[0])):
            element = gray[i][j]
            count = count + 1
            sum = sum + math.pow((element - avg), 2)
    c = sum / (count - 1)
    return c

# 获取图片清晰度（锐度）
def get_sharpness_feature(path):
    kernel_size = 3
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    img = cv2.imread(path)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_lap = cv2.Laplacian(gray, ddepth, ksize=kernel_size, scale=scale, delta=delta)
    s = np.max(np.array(gray_lap))
    return s

'''
https://github.com/leomauro/image-entropy
'''

# 获取图片墒

def shannon_entropy(img):
    histogram = img.histogram()
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])



def getdict(file):
    photo = {}
    with open(file) as f:
        for line in f:
            pid = line.strip().split(',')[0]
            pidx = line.strip().split(',')[1]
            photo[pid] = pidx
    return photo


# 获取LBP59维向量
def blogLBP(image):
    im = cv2.imread(image)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    no_points = 8
    radius = 2
    lbp = local_binary_pattern(im_gray, no_points, radius, method='nri_uniform')  # nri_uniform' is non-rotation invariant uniform and will give 59 dimensions vector,
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    return hist



# 读取图片
def readimg(pid,img):
    # imgs_path = 'data/photos/'
    # img = imgs_path + str(pid) + '.jpg'
    scores = []

    print(img,"===================")

    bright_socre = get_brightness_feature(img)
    saturation_score = get_saturation_feaure(img)
    sharpness_score = get_sharpness_feature(img)
    colorfull_score = get_colorfulness_feature(img)
    natural_score = get_naturalness_feature(img)
    contrast_score = get_contrast_feature(img)
    imge = Image.open(img)
    entropy_score = shannon_entropy(imge)
    lbpvec = blogLBP(img)
    lbpvec = [i for i in lbpvec]

    scores.extend([pid])
    scores.extend([bright_socre,saturation_score,sharpness_score,colorfull_score,natural_score,contrast_score,entropy_score])
    scores.extend(lbpvec)

    return scores

def job(pid):
    imgs_path = 'data/photos/'
    img = str(pid) + '.jpg'
    wf = open('true_sample_pidx_attrac', 'a')
    if os.path.exists(imgs_path + img):
        scores = readimg(pid)
        wf.write(str(scores)+'\n')
    else:
        print(img,"not exits !!!!!")
    wf.close()

def multicore(data):
    pool = Pool()
    res = pool.map(job, data)
    return res

if __name__ == '__main__':

    # true_pidx_file = "data/true_sample_photo_info"
    #
    # piddata = open(true_pidx_file)
    # pidset = [row.strip().split(',')[0] for row in piddata]
    #
    # data = multicore(pidset)
    #
    # print("Finised!!!!")

    data_files = 'data/allimages/'
    wf = open('data/pid_attrac', 'a')
    for root,dirs,files in os.walk(data_files):
        # pid = files.strip('.png')
        for file in files:
            file_path = os.path.join(root,file)
            pid = file[:-4]
            # print(pid)
            # print(file_path)
            scores = readimg(pid,file_path)
            wf.write(str(scores) + '\n')
        # print(files)




