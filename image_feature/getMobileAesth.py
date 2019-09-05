import os
from Nimamodel import Nima
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.models import Model
from keras.preprocessing import image

def getModel(base_model_name, weights_file):

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    model = Model(inputs=nima.nima_model.input, outputs=nima.nima_model.get_layer('global_average_pooling2d_1').output)

    return model

def getAesthVe(pid,img,model):
    print(img, "===================")
    scores = []
    img_f = image.load_img(img, target_size=(224, 224))
    x = image.img_to_array(img_f)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 去均值中心化，preprocess_input函数详细功能见注

    # get predictions
    # predictions = predict(nima.nima_model, image)
    mobile_features = model.predict(x)
    mobil_vec = mobile_features.tolist()

    scores.extend([str(pid)])
    for i in mobil_vec:
        scores.extend(i)

    return scores



if __name__ == '__main__':
    base_model_name = "MobileNet"
    weights_file = "MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"

    model = getModel(base_model_name,weights_file)

    data_files = 'data/allimages/'
    wf = open('data/pid_Aesth', 'a')
    for root,dirs,files in os.walk(data_files):

        for file in files:
            file_path = os.path.join(root,file)
            pid = file[:-4]
            # print(pid)
            # print(file_path)
            scores = getAesthVe(pid,file_path,model)
            wf.write(str(scores) + '\n')

    print("Finised!!!!")

