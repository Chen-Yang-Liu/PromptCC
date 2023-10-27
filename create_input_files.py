
import time
import os
import numpy as np
from imageio import imread
from skimage.transform import resize as imresize
from random import seed, choice, sample
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse, random


def create_input_files(args,dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_changeflag = []

    val_image_paths = []
    val_image_captions = []
    val_image_changeflag = []

    test_image_paths = []
    test_image_captions = []
    test_image_changeflag = []

    # word_freq = Counter()  # 创建一个空的Counter类(计数

    train_num=0
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            # word_freq.update(c['tokens'])   # 其中c['tokens']是一个很多单词组成的句子‘列表’
            # if len(c['tokens']) <= max_len:
            captions.append(c['raw'].replace(' .','').replace('.',''))

        if len(captions) == 0:
            continue

        if dataset == 'LEVIR_CC':
            path1 = os.path.join(image_folder, img['split'], 'A', img['filename'])
            path2 = os.path.join(image_folder, img['split'], 'B', img['filename'])
            path = [path1, path2]
            changeflag = img['changeflag']

        if img['split'] in {'train'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_changeflag.append(changeflag)
            # train_num = train_num+1
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_changeflag.append(changeflag)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_changeflag.append(changeflag)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)


    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img'

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    device = torch.device('cuda:0')

    for impaths, imcaps, imchangeflag, split in [(train_image_paths, train_image_captions, train_image_changeflag, 'TRAIN'),
                                   (val_image_paths, val_image_captions, val_image_changeflag, 'VAL'),
                                   (test_image_paths, test_image_captions, test_image_changeflag, 'TEST')]:

        out_path = os.path.join(output_folder, split +'_' + base_filename + '.pkl')

        feature_list = []
        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):
            # if 'the scene is the same as before' in imcaps[i] and split == 'TRAIN':
            #     continue
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                if split == 'TRAIN':
                    # for nochanged image pairs, just use one kind of nochanged captions during the training
                    if 'the scene is the same as before' in imcaps[i]:
                        imcaps[i] = []
                        imcaps[i].append('the scene is the same as before')
                        imcaps[i] = imcaps[i]*5
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            if dataset =='LEVIR_CC':
                ori_img_A = io.imread(impaths[i][0])
                ori_img_B = io.imread(impaths[i][1])
                images = {'ori_img': [ori_img_A, ori_img_B], 'changeflag': imchangeflag[i]}
            else:
                print("Error")

            feature_list.append(images)

            for j, c in enumerate(captions):
                # Find caption lengths
                c_len = len(c.split(' '))
                enc_c = c+'.'
                enc_captions.append(enc_c)
                caplens.append(c_len)

        # Sanity check
        assert len(feature_list) * captions_per_image == len(enc_captions) == len(caplens)

        with open(out_path, 'wb') as f:
            pickle.dump({"images": feature_list, "captions": enc_captions, 'caplens': caplens}, f)

if __name__ == '__main__':

    print('create_input_files START at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()

    create_input_files(args, dataset='LEVIR_CC',
                       karpathy_json_path=r'./data/LEVIR_CC/LevirCCcaptions_v1.json',
                       image_folder=r'./data/LEVIR_CC/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=r'./data/LEVIR_CC',
                       max_len=50)

    print('create_input_files END at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))