from sklearn.metrics import accuracy_score
import shutil
import os, sys, time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict

sys.path.append(os.getcwd()[:-3] + 'lib')
from datasets import timeSeriesDataset, FewShotSampler
from configs import get_parser, Logger, time_string, convert_secs2time, AverageMeter, obtain_accuracy
from models import euclidean_dist
import models
import datetime
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score
import shutil
import os, sys, time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict

sys.path.append(os.getcwd()[:-3] + 'lib')
from datasets import timeSeriesDataset, FewShotSampler
from configs import get_parser, Logger, time_string, convert_secs2time, AverageMeter, obtain_accuracy
from models import euclidean_dist
import models
import datetime
from sklearn.neighbors import KNeighborsClassifier


def load_SFA_train_data(name):
    args_dataset_root = '/home/tangw/Desktop/TSC/BOSS_NN/SFA_Python-master/test/BOSS_feature_Data_pyts/'
    train_dataset = timeSeriesDataset(args_dataset_root, 'train', name, 1, 10)
    return train_dataset


def random_choice(s_idxs,k):
    result=np.random.choice(s_idxs.size(0), k, replace=False)
    return s_idxs[result]
    


def get_log_path(name, ratio_number, ind_number):
    log_path = os.getcwd()[:-3] + 'logs/5shot_linear_transform/' + name + '/' + str(ratio_number) + '/' + str(
        ind_number)
    log_path = log_path + '/' + os.listdir(log_path)[0]
    log_path = log_path + '/baseline_classifier/model_linear_transform_lst.pth'
    # log_path = log_path+'/baseline_classifier/model_linear_transform_best.pth'
    # log_path = log_path+'/baseline_classifier/info-last-linear_transform.pth'
    return log_path



def save_to_file(sentence, dataset_name, log_path=None):
    father_path = './test_accuracy_log_all_as_proto/' + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + 'NN_list_log.txt'
    if log_path != None:
        path = log_path
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')
    return path


name_list=[
    'ECG200',
  ]
ratio_number_list = [1]
ind_number_list = [10]
args_arch = 'linear_transform'
args_dataset_root = '/home/tangw/Desktop/TSC/BOSS_NN/SFA_Python-master/test/BOSS_feature_Data_pyts/'
using_proto = True
pick_all =True

for name in name_list:
    for ind_radio, ratio_number in enumerate(ratio_number_list):
        for ind, ind_number in enumerate(ind_number_list):
                model_lst_path = get_log_path(name, ratio_number, ind_number)
                
                train_dataset = timeSeriesDataset(args_dataset_root, 'train', name, ratio_number, ind_number)
                model = models.__dict__[args_arch](train_dataset.fea_dim,256,64)
                model = torch.nn.DataParallel(model).cuda()

                checkpoint = torch.load(model_lst_path)
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                
                feas = train_dataset.feature
                labels = train_dataset.label
                embs = model(feas)
                
                cpu_labels = labels.cpu().tolist()
                idxs_dict = defaultdict(list)
                for i, l in enumerate(cpu_labels):
                    idxs_dict[l].append(i)
                idxs_dict = dict(sorted(idxs_dict.items()))
                grouped_s_idxs = []
                for lab, idxs in idxs_dict.items():
                    print(lab)
                    grouped_s_idxs.append(torch.LongTensor(idxs[:]))
                proto_lst = [torch.mean(embs[s_idxs], dim=0) for s_idxs in grouped_s_idxs]
                proto = torch.stack(proto_lst, dim=0)
                logits = - euclidean_dist(embs, proto, transform=True).view(labels.size(0), len(proto))
                dis = logits.cpu().detach().numpy()
             
                print(dis)
                labind= 0
                result = []
                for s_idxs in grouped_s_idxs:
                    temp = dis[s_idxs]
                    print(labind)
                    result.append(s_idxs[np.argmax(temp,axis = 0)[labind]].numpy())
                    labind = labind + 1

                print(labels)
                print(result)
                sentence = ''
                for i in result:
                    sentence = sentence+str(i)+'\t'
                save_to_file(sentence,name)

