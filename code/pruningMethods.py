import math
import pickle
import sys
import time
import copy
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import wandb
import torch.nn.utils.prune as prune

from utils import read_corpus, batch_iter, LabelSmoothingLoss
from vocab import Vocab, VocabEntry

def get_layers(model):
    arr =[]
    for i,j in model.named_parameters():
        a = i.split('.')
        arr.append(tuple(a))
        
    layers = []
    for name, weight in arr:
        for i,j in model.named_children():
            if i == name:
                layers.append([j,weight])
    return layers

class SNIP():
    def __init__(self):
        pass
    
    def get_mask(self, model):
        layers = get_layers(model)
        prune.global_unstructured(layers,pruning_method=prune.RandomUnstructured,amount=0.0)
        
    def set_grad(self, model):
        for name, vector in model.named_buffers():
            if name == 'label_smoothing_loss.one_hot':
                continue
            vector.requires_grad = True
            
    def set_grad_back(self, model):

        self.final_scores = []
        returned_scores = []
        for name, vector in model.named_buffers():
            print(vector.shape,name)
            if name == 'label_smoothing_loss.one_hot':
                continue
            self.final_scores.append(torch.clone(vector.grad).detach().abs_())
            returned_scores.append(self.final_scores[-1].flatten())
            vector.grad.zero_()
            vector.requires_grad = False

        for name, vector in model.named_parameters():
            vector.grad.zero_()

        return torch.cat(returned_scores)
    
    def thresholding(self, model, percent):
        percent = float(percent)
        self.thresh = torch.topk(self.scores, int(self.scores.shape[0]*(percent)), largest=False, sorted=True)
        thresh = self.thresh[0][-1]
        print(thresh)
        
        for score, (name, vector) in zip(self.final_scores, model.named_buffers()):
            if name == 'label_smoothing_loss.one_hot':
                continue
            vector[score <= thresh] = 0
        
    def prune(self, model, data, batches, batch_size, percent, device):
        batches=int(batches)
        self.get_mask(model)
        self.set_grad(model)
        percent=float(percent)
        for idx, (data, target) in enumerate(batch_iter(data, batch_size=batch_size, shuffle=True)):
#             data, target = data.to(device), target.to(device)
            
            loss= model(data, target).sum()
            print(loss)
            loss.backward()
            print(idx)
            if idx+1 == batches:
                break
        
        self.scores = self.set_grad_back(model)
        print(self.scores)
        self.thresholding(model, percent)
        
class OBD():
  def __init__(self):
      pass

  def get_mask(self, model):
      layers = get_layers(model)
      prune.global_unstructured(layers,pruning_method=prune.RandomUnstructured,amount=0.0)

  def set_grad_back(self, model):

      self.final_scores = []
      returned_scores = []
      for name, vector in model.named_parameters():
          print(vector.shape,name)
          if name == 'label_smoothing_loss.one_hot':
              continue
          self.final_scores.append(torch.clone(vector.grad).detach()**2 * vector**2 *(0.5))
          returned_scores.append(self.final_scores[-1].flatten())
          vector.grad.zero_()

      return torch.cat(returned_scores)

  def thresholding(self, model, percent):
      percent = float(percent)
      self.thresh = torch.topk(self.scores, int(self.scores.shape[0]*(percent)), largest=False, sorted=True)
      thresh = self.thresh[0][-1]
      print(thresh)
      with torch.no_grad():
        for score, (name, vector) in zip(self.final_scores, model.named_buffers()):
            if name == 'label_smoothing_loss.one_hot':
                continue
            # print(vector,type(score))
            vector[score <= thresh] = 0

  def prune(self, model, data, batches, batch_size, percent, device):
      batches=int(batches)
      # batches = 1
      self.get_mask(model)
      percent=float(percent)
      for idx, (data, target) in enumerate(batch_iter(data, batch_size=batch_size, shuffle=True)):

          
          loss= model(data, target).sum()
          print(loss)
          loss.backward()
          print(idx)
          if idx+1 == batches:
              break
      
      self.scores = self.set_grad_back(model)
      print(self.scores)
      self.thresholding(model, percent)
            

