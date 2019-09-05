import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
import os


#【0】ResNet50模型，加载预训练权重
base_model = ResNet50(weights="imagenet")
print(base_model.summary())

#【1】创建一个新model, 使得它的输出(outputs)是 ResNet50 中任意层的输出(output),通过打印模型结构可以找到每层的名字，这里我们选择最后一个全连接层1000维
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1000').output)


#【2】输入文件路径，遍历文件夹下所有图片
img_path = 'data/test/'
img_list = os.listdir(img_path)
for img in img_list:
    img_f = image.load_img(img_path+img, target_size=(224, 224)) # 加载图片并resize成224x224

#【3】将图片转化为4d tensor形式
    x = image.img_to_array(img_f)
    x = np.expand_dims(x, axis=0)

#【4】数据预处理
    x = preprocess_input(x) #去均值中心化，preprocess_input函数详细功能见注

#【5】提取特征
    block4_pool_features = model.predict(x)
    print(img,"\n","-------------------------")
    print(block4_pool_features)