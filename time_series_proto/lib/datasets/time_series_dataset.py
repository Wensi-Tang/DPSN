import random
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from collections import defaultdict
import sys
class timeSeriesDataset(data.Dataset):
  def __init__(self, dataset_dir, mode, name, ratio_number, ind_number):
    super(timeSeriesDataset, self).__init__()
    self.dataset_dir  = Path(dataset_dir)
    save_path   = self.dataset_dir / name/ str(ratio_number) / str(ind_number) / "{}.npy".format(name) 
    dictionary  = np.load(save_path)
    self.feature= torch.FloatTensor( dictionary.item().get('X_{}_feature'.format(mode)).todense() )
    self.label = torch.LongTensor( dictionary.item().get('y_{}'.format(mode)) )
    self.boss_accuracy = dictionary.item().get('Boss_accuracy')
    self.n_classes = len(set(self.label.tolist()))
    self.fea_dim   = self.feature.shape[-1]
    print('Building timeSeriesDataset for [{}] [{}] with {} classes ...'.format(name, mode, self.n_classes))
    print("BOSS ACC IS {}".format(self.boss_accuracy))

  def __getitem__(self, idx):
    feature = self.feature[idx] 
    label   = self.label[idx]
    return feature, label

class FewShotSampler(object):
  def __init__(self, label, sample_per_class, iterations):
    self.label            = label
    self.sample_per_class = sample_per_class
    self.all_classes      = list(set(label.tolist())) 
    self.iterations       = iterations


  def __iter__(self):
    for it in range(self.iterations):
      spc = self.sample_per_class
      batch_size = spc * len(self.all_classes) 
      few_shot_batch = []
      for i, c in enumerate(self.all_classes):
        fea_idxs = ( self.label == c).nonzero()[:,0].tolist()
        if len(fea_idxs)<spc:
            few_shot_batch.extend( random.sample(fea_idxs, max(len(fea_idxs),2)))
        else:
            few_shot_batch.extend( random.sample(fea_idxs, spc))
      batch = torch.LongTensor(few_shot_batch)
      yield batch

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations
