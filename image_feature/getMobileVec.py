from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import os
import numpy as np


def readimg(pid,img,model):
    print(img, "===================")
    scores = []
    img_f = image.load_img(img, target_size=(224, 224))
    x = image.img_to_array(img_f)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 去均值中心化，preprocess_input函数详细功能见注
    mobile_features = model.predict(x)
    mobil_vec = mobile_features.tolist()
    # print(mobil_vec)

    scores.extend([str(pid)])
    for i in mobil_vec:
        scores.extend(i)

    return scores




def job(pid,model):
    imgs_path = 'data/photos/'
    img = str(pid) + '.jpg'

    wf = open('pidx_mobile_cnn', 'a')
    if os.path.exists(imgs_path + img):
        scores = readimg(imgs_path + img, pid,model)
        # print(scores)
        wf.write(str(scores)+'\n')
    else:
        print(img,"not exits !!!!!")
    wf.close()


if __name__ == '__main__':
    base_model = ResNet50(weights="imagenet")
    print(base_model.summary())
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)


    data_files = 'data/allimages/'
    wf = open('data/pid_Resnet', 'a')
    for root,dirs,files in os.walk(data_files):

        for file in files:
            file_path = os.path.join(root,file)
            pid = file[:-4]
            # print(pid)
            # print(file_path)
            scores = readimg(pid,file_path,model)
            wf.write(str(scores) + '\n')




    print("Finised!!!!")
