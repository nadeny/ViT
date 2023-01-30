
from tqdm import tqdm
import wandb
import glob
from PIL import Image
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from model import ViT
from utils import LabelSmoothing

import warnings
warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ViT(
            patches=(16, 16),
            d_model=768,
            d_ff=3072,
            num_heads=12,
            num_layers=12,
            dropout=0.1,
            image_size=(3, 384, 384),
            num_classes=1000,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = LabelSmoothing(smoothing=0.1).to(self.device)  # Label smoothing
        self.criterion_val = nn.CrossEntropyLoss().to(self.device)

        self.best_val_loss = float('inf')

    def train(self, train_iter, val_iter, epochs, length, is_log):
        train_len, val_len = length
        if is_log:
            wandb.init(project="Image Classification", entity="na_deny")
            wandb.config = {'Dataset': 'ImageNet'}

        for epoch in range(epochs):
            print("=============================== Epoch: ", epoch + 1, " of ", epochs,
                  "===============================")
            print('\n')
            batch_loss = 0
            batch_acc = 0
            self.model.train()
            for image, target in tqdm(train_iter):
                image = image.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output, _ = self.model(image)

                loss = self.criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                batch_loss += loss.item()
                batch_acc += self._acc(target, output)

            train_loss = batch_loss / len(train_iter)
            train_acc = batch_acc / train_len

            val_loss, val_acc = self.evaluate(val_iter)
            val_acc = val_acc / val_len

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_param()
            if is_log:
                wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

            print(f'\rTrain Loss: {train_loss:.3f} | Train Acc.: {train_acc: .3f} |'
                  f'Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc: .3f}'
                  , end='', flush=True)
            print('\n ')

    @torch.no_grad()
    def evaluate(self, valid_iter):
        self.model.eval()
        batch_loss = 0
        valic_acc = 0
        for image, target in tqdm(valid_iter):
            image = image.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output, _ = self.model(image)
            loss = self.criterion_val(output, target)

            batch_loss += loss.item()
            valic_acc += self._acc(target, output)
        valid_loss = batch_loss / len(valid_iter)
        return valid_loss, valic_acc  # valid_acc is un-normalized True Positive

    @torch.no_grad()
    def inference(self, visualize):
        self.model.load_state_dict(torch.load('models.pth'))
        self.model.eval()

        # Load class names
        label_path = os.path.join('inference_image/labels_map.txt')
        labels_map = json.load(open(label_path))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        for f in glob.iglob('inference_image/*.jpg'):
            print(f'path: {f} | Prediction Result: ')
            im = Image.open(f)
            tfms = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
            img = tfms(im).unsqueeze(0)

            img = img.to(self.device)
            with torch.no_grad():
                outputs, scores = self.model(img)
                outputs = outputs.squeeze(0)


            if visualize:
                scores = torch.stack(scores).squeeze(1)
                scores = scores.to('cpu')
                # Average the attention weights across all heads.
                scores = torch.mean(scores, dim=1)

                # To account for residual connections, we add an identity matrix to the
                # attention matrix and re-normalize the weights.
                residual_score= torch.eye(scores.size(1))
                aug_score = scores + residual_score
                aug_score = aug_score / aug_score.sum(dim=-1).unsqueeze(-1)

                # Recursively multiply the weight matrices
                joint_attentions = torch.zeros(aug_score.size())
                joint_attentions[0] = aug_score[0]

                for n in range(1, aug_score.size(0)):
                    joint_attentions[n] = torch.matmul(aug_score[n], joint_attentions[n - 1])

                # Attention from the output token to the input space.
                v = joint_attentions[-1]
                grid_size = int(np.sqrt(aug_score.size(-1)))
                mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
                mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
                result = (mask * im).astype("uint8")

                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

                ax1.set_title('Original')
                ax2.set_title('Attention Map')
                _ = ax1.imshow(im)
                _ = ax2.imshow(result)
                plt.show()

            print("Prediction Label and Attention Map!\n")
            outputs = outputs.squeeze(0)
            for idx in torch.topk(outputs, k=5).indices.tolist():
                prob = torch.softmax(outputs, -1)[idx].item()
                print('[{idx}]\t {label:<75} ({p:.3f}%)'.format(idx=idx, label=labels_map[idx], p=prob * 100))
            print('\n')



    def _acc(self, target, output):
        acc = 0
        out = torch.argmax(output, dim=1)
        for i in range(target.shape[0]):
            if target[i] == out[i]:
                acc += 1
        return acc

    def save_param(self):
        param = {'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'val_loss': self.best_val_loss}
        torch.save(param, 'models.pt')

    def load_param(self):
        param = torch.load('models.pt')
        self.model.load_state_dict(param['model_state'])
        self.optimizer.load_state_dict(param['optim_state'])
        self.best_val_loss = param['val_loss']




