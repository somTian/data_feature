import os
import glob
from Nimamodel import Nima



def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)




def main(base_model_name, weights_file, image):

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    # predictions = predict(nima.nima_model, image)
    score = nima.nima_model.predict(image)

    return score




if __name__ == '__main__':
    base_model_name = "MobileNet"
    weights_file = "weights/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"

    img_path = 'data/test/'
    img_list = os.listdir(img_path)
    for img in img_list:
        nimavec = main(base_model_name, weights_file, img_path+img)
        print(nimavec)
