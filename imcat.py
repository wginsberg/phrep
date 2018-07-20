#!/usr/bin/env python
import argparse
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications import mobilenet
from keras.applications.imagenet_utils import decode_predictions

def main(args):

    model = mobilenet.MobileNet(weights='imagenet')

    predictions = decode_predictions(
                    model.predict(
                        np.array([resize(
                            imread(args.file),
                            (224, 224),
                            mode='constant')])
                        )
                    )

    for p in predictions[0]:
        print(p[1])

    return predictions

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file')

    args = parser.parse_args()
    main(args)

