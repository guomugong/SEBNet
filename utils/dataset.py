import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import copy

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name):
        # 初始化函数，读取所有data_path下的图片
        self.dataset_name = dataset_name
        self.data_path = data_path

        if self.dataset_name == "ddr":
            self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.jpg'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "idrid":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        if self.dataset_name == "fgadr":
            self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
            self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))

        self.is_train = is_train

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        if self.dataset_name == "ddr":
            label_path = image_path.replace('img', 'label')
            label_path = label_path.replace('.jpg', '_ALL.tif') 
        if self.dataset_name == "idrid":
            label_path = image_path.replace('image', 'label')
            label_path = label_path.replace('.tif', '_ALL.tif') 
        if self.dataset_name == "fgadr":
            label_path = image_path.replace('image', 'label')
            label_path = label_path.replace('.png', '.tif') 

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')

#        if self.is_train == 0: # TEST
#            if self.dataset_name == "ddr":
#
#                crop_size = 640
#                w = (image.size[0]-crop_size)//2 # Central Crop
#                h = (image.size[1]-crop_size)//2
#                image = TF.crop(image, w, h, crop_size, crop_size)
#                label = TF.crop(label, w, h, crop_size, crop_size)
#

        if self.is_train == 1:

            #if np.random.random_sample() <= 0.5:
            #    image = TF.adjust_brightness(image, brightness_factor=np.random.uniform(0.5, 1.5))
            #    image = TF.adjust_contrast(image, contrast_factor=np.random.uniform(0.5, 1.5))
            #    image = TF.adjust_saturation(image, saturation_factor=np.random.uniform(0, 1.5))

            if np.random.random_sample() <= 0.5:
                image, label = self.randomRotation(image, label)

            if np.random.random_sample() <= 0.25:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random_sample() <= 0.25:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

            if np.random.random_sample() <= 1:
                # random.uniform(),从一个均匀分布[low,high)中随机采样
                crop_size = 256
                w = random.uniform(0, image.size[0]-crop_size)
                h = random.uniform(0, image.size[1]-crop_size)
                image = TF.crop(image, w, h, crop_size, crop_size)
                label = TF.crop(label, w, h, crop_size, crop_size)

        image = np.asarray(image)
        label = np.asarray(label)

        image = image.transpose(2, 0, 1)
        label = label.reshape(label.shape[0], label.shape[1])
        label = np.array(label)

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif


        return image, label, filename

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        #random_angle = np.random.randint(1, 360)
        #return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)
        random_angle = np.random.randint(1, 4)
        return image.rotate(random_angle*90, mode), label.rotate(random_angle*90, Image.NEAREST)

    def padding_image(self,image, label, pad_to_h, pad_to_w):
        #新建长宽608像素，背景色为（0, 0, 0）的画布对象,即背景为黑，RGB是彩色三通道图像，P是8位图像
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        # 把新建的图像粘贴在原图上
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        return new_image, new_label
