# -*- coding: utf-8 -*-
"""Char generation PyTorch.ipynb
"""

from pathlib import Path

import requests


args = {
    # for model
    'batch_size': 512,
    'test_batch_size': 128,
    'epochs':200,
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


import string

# TODO: make these not be globals
CHARS=string.printable
NUM_CLASSES=len(CHARS)

class CharModel(pl.LightningModule):
    def __init__(self, lr=.01):
        super().__init__()
        self.hidden_size = args["rnn_hidden_size"]

        
        self.lstm = nn.LSTM(input_size=NUM_CLASSES, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, NUM_CLASSES) 
        self.truncated_bptt_steps = args['truncated_bptt_steps']
        self.learning_rate = lr

    # Return the hidden tensor(s) to pass to forward
    def getNewHidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=self.device),
                torch.zeros(1, batch_size, self.hidden_size,device=self.device))

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

    def validation_step(self, batch, batch_idx):
        data, y = batch
        hidden = self.getNewHidden(batch_size=data.shape[0])
        y_hat, hidden = model(data, hidden)
        c, t = self.char_accuracy(y_hat, y)
        # yhat has dimension: batch, seq, C
        # Need batch, C, seq
        loss = F.cross_entropy(y_hat.transpose(1, 2),  y)
        self.log("val_loss", loss)
        self.log("val_accuracy", 100. * c / t)
        if batch_idx == 0:
            sample = self.generateUnconditionally()
            self.logger.experiment.add_text('sample', sample, self.current_epoch)


    def generateUnconditionally(self, prompt=args["prompt"],
            temperature=args["temperature"],
            output_length=args["generated_length"]):
        VERBOSE=False
        result = ""
        hidden = self.getNewHidden(batch_size=1)
        char = 0

        # Not currently initializing with a prompt
        #with torch.no_grad():
        #for c in prompt:
            #input=torch.reshape(torch.tensor(ord(c)), (1, 1)).to(device)
            #_, hidden = model.forward(input, hidden)

        for step in range(output_length):
            input=torch.reshape(torch.tensor(char, device=self.device), (1, 1))
            with torch.no_grad():
              predictions, hidden = self.forward(input, hidden)
            #print('predictions.shape', predictions.shape)
            # Only use the last prediction.
            pred = predictions[0, -1, :].cpu()
            #print('pred', pred)
            pred = pred/temperature;
            #print('pred after temperature', pred)
            newChar = torch.distributions.categorical.Categorical(logits=pred).sample()

            result += CHARS[newChar.item()]
            char = newChar
        return result

    def configure_optimizers(self):
        print('self.learning_rate', self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #optimizer,
            #max_lr=self.learning_rate,
            #total_steps=self.trainer.estimated_stepping_batches
        #)
        return [optimizer], []

from torch.utils.data import random_split
from torch.utils.data import DataLoader

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

class HttpTextDataModule(pl.LightningDataModule): 
    def __init__(self,
            url = "https://sherlock-holm.es/stories/plain-text/houn.txt",
            bs = 32):
        super().__init__()
        self.url = url
        self.bs = bs

    def prepare_data(self):
        response = requests.get(self.url)
        text = response.content
        import string
        print(f"text length before: {len(text)}")
        trimmedText = []
        for c in text:
          try:
            trimmedText.append(CHARS.index(chr(c)))
          except:
            pass
        print(f"text length after: {len(trimmedText)}")
        self.dataset = SeqDataset(trimmedText)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.bs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.bs)


    def setup(self, stage = None):
        train_split = int(len(self.dataset)*.8)
        valid_split = len(self.dataset) - train_split
        self.train_ds, self.val_ds = random_split(self.dataset, [train_split, valid_split])



model = CharModel()

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/data/neil/runs", name="char_generation-lightning")

lr_monitor = LearningRateMonitor(logging_interval='step')

data_module = HttpTextDataModule()
trainer = pl.Trainer(progress_bar_refresh_rate=1,
        max_epochs=args["epochs"],
        gpus=1,
        logger=logger,
        callbacks=[lr_monitor],
        #auto_lr_find=True
        )

#trainer.tune(model, datamodule=data_module)

logger.log_hyperparams(args)

trainer.fit(model, datamodule=data_module)    
