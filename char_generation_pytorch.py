# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""

from pathlib import Path

import requests

url = "https://sherlock-holm.es/stories/plain-text/houn.txt"
response = requests.get(url)
text = response.content

args = {
    # for model
    'batch_size': 512,
    'test_batch_size': 128,
    'epochs':10,
    'max_seq_length': 50,
    'truncated_bptt_steps': 5,
    'lr': .001,
    'max_lr': .01,
    'steps_per_epoch': 10,
    'optimizer': 'adam',
    "rnn_hidden_size": 100,
    "rnn_type": "lstm",

    # for generation
    'temperature': 0.9,
    "prompt": "A",
    "generated_length": 50,

    # meta
    'seed': 1,
    'log_interval': 1000,

}


"""# Initialization"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

torch.manual_seed(args["seed"])
use_cuda = torch.cuda.is_available()
print("cuda is available", use_cuda)
device = torch.device("cuda")




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Sampler


"""#Model"""

class CharModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = args["rnn_hidden_size"]
        self.lstm = nn.LSTM(input_size=NUM_CLASSES, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, NUM_CLASSES) 
        self.truncated_bptt_steps = args['truncated_bptt_steps']


    # Return the hidden tensor(s) to pass to forward
    def getNewHidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, hidden):
        VERBOSE=False
        if VERBOSE:
          print(f'Forward: size of input: {x.size()}')

        x = F.one_hot(x, num_classes=NUM_CLASSES).float()
        if VERBOSE:
          print(f'Forward: after one_hot: size of input: {x}')
       
        (x, hidden) = self.lstm(x, hidden)

        if VERBOSE:
          print(f'Forward: size after rnn: {x.size()}')
          #print(f'Forward: after rnn: {x} ')
        x = self.fc(x)
        if VERBOSE:
          print(f'Forward: size after fc: {x.size()}')
          #print(f'Forward:  after fc: {x}')
        return x, hidden
    
    def char_accuracy(self, output, target):
      mostLikely = torch.argmax(output, dim=2)
      #rint(f"mostLikely size: {mostLikely.size()}")
      #print(f"target size: {target.size()}")
      eq = mostLikely.eq(target.view_as(mostLikely))
      #print(f"eq: {eq.size()}, {eq}")
      #print(f"eq.sum(): {eq.sum().size()}, {eq.sum()}")
      correct = eq.sum().item()
      total = torch.numel(eq)
      #print(f"correct, total: {correct}, {total}")
      return correct, total

    def training_step(self, batch, batch_idx, hiddens):
        data, y = batch
        print(f"data.shape {data.shape}")
        y_hat, hiddens = self(data, hiddens)
        
        correct, total = self.char_accuracy(y_hat, y)

        # yhat has dimension: batch, seq, C
        # Need batch, C, seq
        loss = F.cross_entropy(y_hat.transpose(1, 2),  y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', 100.*correct/total)

        return {"loss": loss, "hiddens": hiddens}

    #def validation_step(self, batch, batch_idx):
        #data, y = batch
        #hidden = 
        #y_hat, hidden = model(data, hidden)
        ##c, t = char_accuracy(output, target)
        ## yhat has dimension: batch, seq, C
        ## Need batch, C, seq
        #loss = F.cross_entropy(y_hat.transpose(1, 2),  y)
        #return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args["lr"])

"""Create dataset class for our data. """

# Break up given data into chunks of max_seq_length each.
# TODO(neil): make this random sequences rather than fixed
class SeqDataset(Dataset):
  def __init__(self, data, seq_length=args['max_seq_length']):
      """data is a byte array"""
      self.seq_length = seq_length
      self.num_sequences = (len(data) - self.seq_length) // self.seq_length
      self.data = np.array(data, dtype=np.uint8)

  def __len__(self):
      return (len(self.data) - self.seq_length) // self.seq_length

  def __getitem__(self, idx):
      #print(f'{idx*self.seq_length}:{(idx+1)*self.seq_length+1}')
      t = torch.as_tensor(self.data[idx*self.seq_length:(idx+1)*self.seq_length+1]).long()
      #print(t)
      return t[:-1], t[1:]

import string
CHARS=string.printable
NUM_CLASSES=len(CHARS)
print(f"text length before: {len(text)}")
trimmedText = []
for c in text:
  try:
    trimmedText.append(CHARS.index(chr(c)))
  except:
    pass
print(f"text length after: {len(trimmedText)}")
from torch.utils.data import random_split
from torch.utils.data import DataLoader

dataset = SeqDataset(trimmedText)
train_split = int(len(dataset)*.8)
valid_split = len(dataset) - train_split
train_ds, val_ds = random_split(dataset, [train_split, valid_split])

train_loader = DataLoader(train_ds, batch_size=32)
val_loader = DataLoader(val_ds, batch_size=32)


def char_accuracy(output, target):
  mostLikely = torch.argmax(output, dim=2)
  #rint(f"mostLikely size: {mostLikely.size()}")
  #print(f"target size: {target.size()}")
  eq = mostLikely.eq(target.view_as(mostLikely))
  #print(f"eq: {eq.size()}, {eq}")
  #print(f"eq.sum(): {eq.sum().size()}, {eq.sum()}")
  correct = eq.sum().item()
  total = torch.numel(eq)
  #print(f"correct, total: {correct}, {total}")
  return correct, total
  
model = CharModel()

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/data/neil/runs", name="char_generation-lightning")

trainer = pl.Trainer(progress_bar_refresh_rate=1, max_epochs=2, gpus=1, logger=logger)

logger.log_hyperparams(args)

trainer.fit(model, train_loader, val_loader)    
