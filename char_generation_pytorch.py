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
valid_split = int(len(trimmedText)*.8)
train_samples=trimmedText[:valid_split]
test_samples=trimmedText[valid_split:]
print(len(train_samples), len(test_samples))

"""# Initialization"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(args["seed"])
use_cuda = torch.cuda.is_available()
print("cuda is available", use_cuda)
device = torch.device("cuda")

"""## logging"""
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=Path("/data") / "neil" / "runs" / "char-generation")
layout = {
    "ABCDE": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
    },
}

writer.add_custom_scalars(layout)



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import Sampler

"""Create dataset class for our data. """

# We've got data from 1..n. We don't want to return in ord
class SeqDataset(Dataset):
  def __init__(self, data, bs=args['batch_size'], max_seq_length=args['max_seq_length']):
      """data is a byte array"""
      self.seq_length = max_seq_length
      self.bs = bs
      num_sequences = (len(data) - max_seq_length) // self.seq_length
      self.num_batches = num_sequences // self.bs
      # even multiple into number of batches
      self.num_sequences = self.num_batches * self.bs
      self.data = np.array(data, dtype=np.uint8)

  def __len__(self):
      return self.num_sequences

  def transformIndex(self, idx):
      # if we have sequences 0, 1, 2, 3, 4, 5, 6, 7 and a BS of 2
      # we want to return sequence: 0, 4, 1, 5, 2 ,6, 3, 7
      batchNum = idx // self.bs
      indexInBatch = idx % self.bs
      return indexInBatch * self.num_batches + batchNum

  def __getitem__(self, idx):
      idx = self.transformIndex(idx)
      #print(f'{idx*self.seq_length}:{(idx+1)*self.seq_length+1}')
      t = torch.as_tensor(self.data[idx*self.seq_length:(idx+1)*self.seq_length+1]).long()
      #print(t)
      return t[:-1], t[1:]

trainDataset = SeqDataset(train_samples)
testDataset = SeqDataset(test_samples)

x, y = trainDataset[36]
len(trainDataset), x, y

"""#Model"""

class Net(nn.Module):
    def __init__(self, batch_size=1):
        super(Net, self).__init__()
        self.hidden_size = args["rnn_hidden_size"]
        self.lstm = nn.LSTM(input_size=NUM_CLASSES, hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, NUM_CLASSES) 

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

model = Net().to(device)
out = model(torch.unsqueeze(x, 0).to(device), model.getNewHidden(1))

"""
Measure the binary cross entropy loss between the prediction and the target.


"""

lf = torch.nn.CrossEntropyLoss()

def loss_func(yhat, y):
  #print(f'loss_func: yhat: {yhat.size()}, y: {y.size()}')
  # yhat has dimension: batch, seq, C
  # Need batch, C, seq
  yhat = yhat.transpose(1, 2)
  #print(f'loss_func after transpose: yhat: {yhat.size()}, y: {y.size()}')
  return lf(yhat, y)

"""# Train"""

def generateUnconditionally(prompt=args["prompt"], temperature=args["temperature"], output_length=args["generated_length"]):
    VERBOSE=False
    model.eval();
    result = ""
    hidden = model.getNewHidden(batch_size=1)
    char = 0

    # Not currently initializing with a prompt
    #with torch.no_grad():
        #for c in prompt:
            #input=torch.reshape(torch.tensor(ord(c)), (1, 1)).to(device)
            #_, hidden = model.forward(input, hidden)
    
    for step in range(output_length):
        input=torch.reshape(torch.tensor(char), (1, 1)).to(device)
        with torch.no_grad():
          predictions, hidden = model.forward(input, hidden)
        #print('predictions.shape', predictions.shape)
        # Only use the last prediction.
        pred = predictions[0, -1, :].cpu()
        #print('pred', pred)
        if False:
          _, c = torch.topk(pred, k=k)
          #print(c)
          newChar = np.random.choice(c.tolist())
        else:
          pred = pred/temperature;
          #print('pred after temperature', pred)
          newChar = torch.distributions.categorical.Categorical(logits=pred).sample()

        result += CHARS[newChar.item()]
        char = newChar
    return result

from tqdm.notebook import tqdm, trange

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
  
def train(args, model, device, loader, optimizer, epoch):
    model.train()
    hidden = model.getNewHidden(batch_size=loader.batch_size)
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="batches", leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #print(f"data.shape {data.shape}")
        output, hidden = model(data, hidden)
        
        hidden = (hidden[0].detach(), hidden[1].detach())

        #print(f"output: {output}, requires_grad: {output.requires_grad}")
        #print(f"output.shape: {output.shape}")
        #print(f"target.shape: {target.shape}")
        loss = loss_func(output, target)
        #print(f"loss: {loss}, requires_grad: {loss.requires_grad}")
        correct, total = char_accuracy(output, target)
        loss.backward(retain_graph=True)

        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
          writer.add_scalar('loss/train', loss, epoch)
          writer.add_scalar('accuracy/train', 100.*correct/total, epoch)


def test(model, device, loader, epoch):
    model.eval()
    hidden = model.getNewHidden(batch_size=loader.batch_size)
    test_loss = 0
    correct = 0
    total = 0
    val_losses = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            #print(f"test: data.shape: {data.shape}")
            output, hidden = model(data, hidden)
            val_losses.append(loss_func(output, target).item())
            c, t = char_accuracy(output, target)
            correct += c
            total += t

    acc = 100. * correct / total
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, total,
    #    acc))
    # generate a sample of handwriting
    #stroke = generateUnconditionally()
    #plot_stroke(stroke, save_name="output.png")


    sample = generateUnconditionally()
    writer.add_scalar('loss/val', np.mean(val_losses), epoch)
    writer.add_scalar('accuracy/val', acc, epoch)
    return test_loss


train_kwargs = {'batch_size': args["batch_size"], 'drop_last': True}
test_kwargs = {'batch_size': args["test_batch_size"], 'drop_last': True}
if use_cuda:
    cuda_kwargs = { 'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_loader = torch.utils.data.DataLoader(trainDataset,**train_kwargs)
test_loader = torch.utils.data.DataLoader(testDataset, **test_kwargs)

model = Net(batch_size=args["batch_size"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=args["lr"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=args["max_lr"],
    epochs=args["epochs"],
    steps_per_epoch=args["steps_per_epoch"])

import time

start = time.time()
for epoch in trange(1, args["epochs"] + 1, desc="epochs"):
    train(args, model, device, train_loader, optimizer, epoch)
    val_loss = test(model, device, test_loader, epoch)
    scheduler.step(val_loss)

# Commented out IPython magic to ensure Python compatibility.
# %env CUDA_LAUNCH_BLOCKING=1

generateUnconditionally(prompt="Sher", temperature=1, output_length=50)

