import os
import sys
import json
import numpy as np
import time

from PIL import Image, ImageDraw

from cigbuttsconfig import CigButtsConfig
from cocolikedataset import CocoLikeDataset

if __name__ == "__main__":
    ROOT_DIR = '/home/surromind/shmoon/maskrcnn/aktwelve_mask_rcnn/'
    assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'
    sys.path.append(ROOT_DIR)

    from mrcnn.config import Config
    import mrcnn.utils as utils
    from mrcnn import visualize
    import mrcnn.model as modellib
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = CigButtsConfig()
    config.display()

    # Create the training and Validation Dataset

    dataset_train = CocoLikeDataset()
    dataset_train.load_data(ROOT_DIR + 'data/cig_butts/train/coco_annotations.json', 
                            ROOT_DIR + '/data/cig_butts/train/images')
    dataset_train.prepare()

    dataset_val = CocoLikeDataset()
    dataset_val.load_data(ROOT_DIR + 'data/cig_butts/val/coco_annotations.json', 
                           ROOT_DIR + '../datasets/cig_butts/val/images')
    dataset_val.prepare()

    # Create the training model & Train
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


    # Which weights to start with?
    # ToDo --> 이 부분 모듈화해서 따로 빼기
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)


    # Train
    
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    start_train = time.time()
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=4, 
                layers='heads')
    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')