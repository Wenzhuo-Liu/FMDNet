from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os.path as osp
import cv2
from scipy import signal
from torchvision.transforms import Compose, Normalize
import codecs


class Mydata(Dataset):
    def __init__(self,
                 img_path=r'F:\UAH\UAH_DATA\motor_dataset\train\image',
                 txt_path=r'F:\UAH\UAH_DATA\motor_dataset\train\txt',
                 label_path=r'F:\UAH\UAH_DATA\motor\train.txt'):

        self.img_path = img_path

        self.incline_path = txt_path + '\\' + 'incline'
        self.KFX_a_path = txt_path + '\\' + 'KFX_a'
        self.KFY_a_path = txt_path + '\\' + 'KFY_a'
        self.KFZ_a_path = txt_path + '\\' + 'KFZ_a'
        self.roll_path = txt_path + '\\' + 'roll'
        self.speed_path = txt_path + '\\' + 'speed'
        self.Yaw_path = txt_path + '\\' + 'Yaw'



        self.label_path = label_path

        self.img_list = []
        self.txt_list = []
        self.label_list = []

        with codecs.open(self.label_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                _list = i.split()
                self.img_list.append(_list[0])
                self.txt_list.append(_list[1])
                self.label_list.append(_list[2])
        # print(self.img_list) : ['0.jpg', '1.jpg'...]
        # print(self.txt_list) : ['0.txt', '1.txt'...]
        # print(self.label_list) : ['0', '1'...]

        self.img_transform = Compose(
            [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.spectrogram_transform = Compose(
            [Normalize(mean=[0.0], std=[12.0])]
        )

    def generate_spectrogram(self, something_txt_path):
        samples = []
        with codecs.open(something_txt_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                samples.append(i)
        samples = np.array(
            list(map(float, samples))
        )

        frequencies, times, spectrogram = signal.spectrogram(samples, 1260, nperseg=512, noverlap=483)
        if spectrogram.shape != (257, 200):
            print('在生成spectrogram时形状不为257x200, 故返回rand值 - class: Mydata, def generate_spectrogram')
            return torch.Tensor(np.random.rand(1, 257, 200))
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)
        spectrogram = torch.Tensor(spectrogram.reshape(spec_shape))
        spectrogram = self.spectrogram_transform(spectrogram)
        return spectrogram

    def __len__(self):
        return len(
            open(self.label_path, 'r').readlines()
        )

    def __getitem__(self, idx):
        image_path = osp.join(self.img_path, self.img_list[idx])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = self.img_transform(torch.Tensor(image))

        incline_txt_path = osp.join(self.incline_path, self.txt_list[idx])
        KFX_a_txt_path = osp.join(self.KFX_a_path, self.txt_list[idx])
        KFY_a_txt_path = osp.join(self.KFY_a_path, self.txt_list[idx])
        KFZ_a_txt_path = osp.join(self.KFZ_a_path, self.txt_list[idx])
        roll_txt_path = osp.join(self.roll_path, self.txt_list[idx])
        speed_txt_path = osp.join(self.speed_path, self.txt_list[idx])
        Yaw_txt_path = osp.join(self.Yaw_path, self.txt_list[idx])

        incline_spectrogram = self.generate_spectrogram(incline_txt_path)
        KFX_a_spectrogram = self.generate_spectrogram(KFX_a_txt_path)
        KFY_a_spectrogram = self.generate_spectrogram(KFY_a_txt_path)
        KFZ_a_spectrogram = self.generate_spectrogram(KFZ_a_txt_path)
        roll_spectrogram = self.generate_spectrogram(roll_txt_path)
        speed_spectrogram = self.generate_spectrogram(speed_txt_path)
        Yaw_spectrogram = self.generate_spectrogram(Yaw_txt_path)

        # whole_spectrogram = torch.concat((incline_spectrogram, KFX_a_spectrogram, KFY_a_spectrogram,
        #                                   KFZ_a_spectrogram, roll_spectrogram, speed_spectrogram,
        #                                   Yaw_spectrogram), dim=0)

        label = [int(self.label_list[idx])]
        return image, incline_spectrogram, KFX_a_spectrogram, KFY_a_spectrogram,KFZ_a_spectrogram, speed_spectrogram, roll_spectrogram, Yaw_spectrogram, torch.LongTensor(label)



if __name__ == "__main__":

    train_datasets = Mydata()

    data_sample = train_datasets[1]

    print('debugger')

    print(data_sample[8])
