
import os
import PIL
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To avoid error due to truncated image

import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from trainer import Trainer


def run(batch_size=1024, worker=4, epoch=1024, is_log=False, mode='inference'):
    trainer = Trainer()
    if mode == 'train':
        parent_dir = 'D:/Dataset Collection/imagenet'  # Point to your imagenet directory
        traindir = os.path.join(parent_dir, 'train')  # imagenet directory contains train & val subfolder
        valdir = os.path.join(parent_dir, 'val')

        normalize = transforms.Normalize(0.5, 0.5)  # set mean: 0.5, std: 0.5, image range: [-1, 1]

        train_dataset = datasets.ImageFolder(traindir,
                                             transforms.Compose([
                                                 transforms.RandomResizedCrop(384),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize,
                                             ]))  # Flow image from directory

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=worker, pin_memory=True)  # load dataset per batch

        val_transforms = transforms.Compose([transforms.Resize(384, interpolation=PIL.Image.BICUBIC),
                                             transforms.CenterCrop(384),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])

        val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, val_transforms),
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=worker, pin_memory=True)

        length = [1281167, 50000]  # train & test size

        trainer.train(train_iter=train_loader, val_iter=val_loader, epochs=epoch, length=length, is_log=is_log)

    if mode == 'inference':
        trainer.inference(visualize=True)  # put any image inside inference_image folder


if __name__ == '__main__':

    mode = 'inference'  # Mode either 'train' or 'inference'

    batch = 1024  # Set suitable batch number
    epochs = 1000
    workers = 4  # Set worker to load dataset

    run(epoch=epochs, batch_size=batch, worker=workers, is_log=False, mode=mode)
