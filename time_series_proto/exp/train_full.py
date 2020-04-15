import os, sys, time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import timeSeriesDataset, FewShotSampler
from configs import get_parser, Logger, time_string, convert_secs2time, AverageMeter, obtain_accuracy
from models import euclidean_dist
import models
import datetime
from os.path import dirname

def covert_list_to_dic(XX_list):
    result_dic ={}
    for i,item in enumerate(XX_list):
        result_dic[item]=i
    return result_dic

def load_proto_result(name,ratio_number_list, ind_number_list):
    path = dirname(os.getcwd())+'/time_series_proto/test_accuracy_log_all_as_proto/'+name+'/'+name+'_log.txt'
    ratio_dic = covert_list_to_dic(ratio_number_list)
    ind_dic = covert_list_to_dic(ind_number_list)
    proto_result = np.zeros([len(ratio_number_list),len(ind_number_list)])
    try:
        f = open(path, "r")
        for x in f:
            temp = x.split('\t')
            if float(temp[3])==1:
                proto_result[-1:,:] =np.ones((1,len(ind_number_list)))*float(temp[-1]) /100
                continue
            
            proto_result[ratio_dic[float(temp[3])]][ind_dic[float(temp[5])]] = float(temp[-1])/100
    except:
        print(name,'cannot find')
        print('searched path is:',path)
    
    return proto_result








def get_proto(args,model,proto_bag):
    name = proto_bag[0]
    ratio_number = proto_bag[1]
    ind_number = proto_bag[2]
    
    train_dataset    = timeSeriesDataset(args.dataset_root, 'train', name,ratio_number, ind_number)
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
        grouped_s_idxs.append(torch.LongTensor(idxs[:]))
    proto_lst = [torch.mean(embs[s_idxs], dim=0) for s_idxs in grouped_s_idxs]
    proto = torch.stack(proto_lst, dim=0)
    return proto



def save_to_file(sentence, dataset_name, log_path=None):
    father_path = './test_accuracy_log_all_as_proto/' + dataset_name
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    path = father_path + '/' + dataset_name + '_log.txt'
    if log_path != None:
        path = log_path
    with open(path, "a") as myfile:
        myfile.write(sentence + '\n')
    return path




def train_model(lr_scheduler, model, criterion, optimizer, logger, dataloader, epoch, args, mode, n_support, n_query,proto_bag):
  args = deepcopy(args)
  losses, acc1 = AverageMeter(), AverageMeter()
  data_time, batch_time, end  = AverageMeter(), AverageMeter(), time.time()
  num_device = len(model.device_ids)

  if mode == "train":
    model.train()
  elif mode == "test":
    model.eval()
    metaval_accuracies = []
  else: raise TypeError("invalid mode {:}".format(mode))

  for batch_idx, (feas, labels) in enumerate(dataloader):
    if mode=="train":
      lr_scheduler.step()

    cpu_labels = labels.cpu().tolist()
    idxs_dict = defaultdict(list)
    for i, l in enumerate(cpu_labels):
      idxs_dict[l].append(i)
    idxs_dict = dict(sorted(idxs_dict.items()))
    grouped_s_idxs, grouped_q_idxs = [], []
    for lab, idxs in idxs_dict.items():
      grouped_s_idxs.append(torch.LongTensor(idxs[:n_support]))
      grouped_q_idxs.append(torch.LongTensor(idxs[n_support:]))
    query_idxs   = torch.cat(grouped_q_idxs, dim=0).tolist()
 
    embs  = model(feas)
    ## use first n support's mean to get proto (this proto from test dataset?)(why do we only use two?)
    

        
    #proto_lst = [torch.mean( embs[s_idxs], dim=0) for s_idxs in grouped_s_idxs]  
    #proto = torch.stack(proto_lst, dim=0)
    
    proto = get_proto(args,model,proto_bag)

    # classification 
    ## get test featuer after liner transform (data loader random sample?)
    query_emb    = embs[query_idxs] 
    logits       = - euclidean_dist(query_emb, proto, transform=True).view(len(query_idxs), len(proto))
    query_labels = labels[query_idxs].cuda(non_blocking=True)
    loss         = criterion(logits, query_labels)
    losses.update(loss.item(), len(query_idxs))  

    top_fs        = obtain_accuracy(logits, query_labels, (1,))
    acc1.update(top_fs[0].item(),  len(query_idxs))

    if mode == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    elif mode=="test":
      metaval_accuracies.append(top_fs[0].item())
      if batch_idx + 1 == len(dataloader):
        metaval_accuracies = np.array(metaval_accuracies)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(batch_idx + 1)
        logger.print("ci95 is : {:}".format(ci95))
    else: raise ValueError('Invalid mode = {:}'.format( mode ))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    if (mode=="train" and ((batch_idx % args.log_interval == 0) or (batch_idx + 1 == len(dataloader)))) \
    or (mode=="test" and (batch_idx + 1 == len(dataloader))):
      Tstring = 'TIME[{data_time.val:.2f} ({data_time.avg:.2f}) {batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(data_time=data_time, batch_time=batch_time)
      Sstring = '{:} {:} [Epoch={:03d}/{:03d}] [{:03d}/{:03d}]'.format(time_string(), mode, epoch, args.epochs, batch_idx, len(dataloader))
      Astring = 'loss=({:.3f}, {:.3f}), acc@1=({:.1f}, {:.1f})'.format(losses.val, losses.avg, acc1.val, acc1.avg)

      logger.print('{:} {:} {:} \n'.format(Sstring, Tstring, Astring))
  
  return losses, acc1


def run(args, model, logger, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader,proto_bag):
  args = deepcopy(args)
  start_time = time.time()
  epoch_time = AverageMeter()
  arch = args.arch 
  model_best_path = '{:}/model_{:}_best.pth'.format(str(logger.baseline_classifier_dir), arch)
  model_lst_path  = '{:}/model_{:}_lst.pth'.format(str(logger.baseline_classifier_dir), arch)

    
  if os.path.isfile(model_lst_path):
    checkpoint  = torch.load(model_lst_path)
    start_epoch = checkpoint['epoch'] + 1
    best_acc    = checkpoint['best_acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print ('load checkpoint from {:}'.format(model_lst_path))
  else:
    start_epoch = 0   
    best_acc    = 0

  for iepoch in range(start_epoch, args.epochs):

    time_str = convert_secs2time(epoch_time.val * (args.epochs- iepoch), True)
    logger.print ('Train {:04d} / {:04d} Epoch, [LR={:6.4f} ~ {:6.4f}], {:}'.format(iepoch, args.epochs, min(lr_scheduler.get_lr()), max(lr_scheduler.get_lr()), time_str))
    
    train_loss, acc1 = train_model(lr_scheduler,  model, criterion, optimizer, logger, train_dataloader, iepoch, args, 'train', args.num_support_tr, args.num_query_tr,proto_bag)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

    info = {'epoch'           : iepoch,
            'args'            : deepcopy(args),
            'finish'          : iepoch+1==args.epochs,
            'best_acc'        : best_acc,
            'model_state_dict': model.state_dict(),
            'optimizer'       : optimizer.state_dict(),
            'scheduler'       : lr_scheduler.state_dict(),
            }
    torch.save(info, model_lst_path)
    logger.print(' -->> joint-arch :: save into {:}\n'.format(model_lst_path))
    with torch.no_grad():
      if iepoch % args.test_interval == 0 :
        logger.print ('----------------------')
        test_loss, test_acc1 = train_model(None, model, criterion, None, logger, test_dataloader, -1, args, 'test', args.num_support_val, args.num_query_val,proto_bag)
        logger.print ('Epoch: {:04d} / {:04d} || Train-Loss: {:.4f} Train-Acc: {:.3f} || Test-Loss: {:.4f} Test-Acc1: {:.3f}\n'.format(iepoch, args.epochs, train_loss.avg, acc1.avg, test_loss.avg, test_acc1.avg))
        if test_acc1.avg >= best_acc:
          torch.save(info, model_best_path)
          best_acc = test_acc1.avg

  best_checkpoint = torch.load(model_best_path)
  model.load_state_dict( best_checkpoint['model_state_dict'] )
  best_acc = best_checkpoint['best_acc']
  
  with torch.no_grad():
    test_loss, test_acc1 = train_model(None, model, criterion, None, logger, test_dataloader, -1, args, 'test', args.num_support_val, args.num_query_val,proto_bag)
  #logger.print ('*[TEST-Best]* ==> Test-Loss: {:.4f} Test-Acc1: {:.3f}, Test-Acc_base: {:.3f}, the TEST-ACC in record is {:.4f}'.format(test_loss.avg, test_acc1.avg, best_acc))
  logger.print ('*[TEST-Best]* ==> Test-Loss: {:.4f} Test-Acc1: {:.3f}, the TEST-ACC in record is {:.4f}'.format(test_loss.avg, test_acc1.avg, best_acc))

  info_path = os.path.join(str(logger.baseline_classifier_dir), "info-last-{:}.pth".format(args.arch))
  torch.save({'model_state_dict': best_checkpoint['model_state_dict']}, info_path)
  return info_path,test_acc1.avg
  

def main():
  args = get_parser()
  args.dataset_root= os.getcwd()[:-18]+'/SFA_Python-master/test/BOSS_feature_Data_pyts/'
  print(args.dataset_root)
  #SFA_Python-master/test/BOSS_feature_Data_pyts/ 
  # create logger
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
  #logger = Logger(str(args.log_dir)+'\'+name, args.manual_seed)
  #logger.print ("args :\n{:}".format(args))

  assert torch.cuda.is_available(), 'You must have at least one GPU'

  # set random seed
  torch.backends.cudnn.benchmark = True
  np.random.seed(args.manual_seed)
  torch.manual_seed(args.manual_seed)
  torch.cuda.manual_seed(args.manual_seed)

  name_list=[
    'ArrowHead',
    'BME',
    'CBF',
    'Chinatown',
    'ECG200',
    'GunPoint',
    'GunPointAgeSpan',
    'GunPointOldVersusYoung',
    'ItalyPowerDemand',
    'MoteStrain',
    'Plane',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'SyntheticControl',
    'ToeSegmentation1',
    'TwoLeadECG',
    'UMD',
    'Wine',
  ]
  ratio_number_list =  [1]
  ind_number_list = [10]
  for name in name_list:
        for ratio_number_ind, ratio_number in enumerate(ratio_number_list):
           
          for ind_number in ind_number_list:

            #try:
                proto_bag= [name,ratio_number,ind_number]
                
                logger = Logger(str(args.log_dir)+'/'+name+'/'+str(ratio_number)+'/'+str(ind_number), args.manual_seed)
                logger.print ("args :\n{:}".format(args))
                # create dataloader
                train_dataset    = timeSeriesDataset(args.dataset_root, 'train', name,ratio_number, ind_number)
                train_sampler    = FewShotSampler(train_dataset.label, args.num_support_tr + args.num_query_tr, args.iterations)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=args.workers)

                test_dataset     = timeSeriesDataset(args.dataset_root, 'test', name,ratio_number, ind_number)
                test_sampler     = FewShotSampler(test_dataset.label, args.num_support_tr + args.num_query_tr, 600)
                test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=args.workers)
                
                # create model
                model = models.__dict__[args.arch](train_dataset.fea_dim,256,64)
                model  = torch.nn.DataParallel(model).cuda()
                logger.print ("model:::\n{:}".format(model))
                
                criterion = nn.CrossEntropyLoss().cuda()
                params = [p for p in model.parameters()] 
                optimizer    = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=args.lr_gamma, step_size=args.lr_step)
                info_path ,test_acc1_avg= run(args, model, logger, criterion, optimizer, lr_scheduler, train_dataloader, test_dataloader,proto_bag)
                logger.print ('save into {:}'.format(info_path))
                sentence = 'dataset_name=\t'+name+'\t'+'ratio=\t'+str(ratio_number)+'\t'+'ind=\t'+str(ind_number)+'\t'+'test_acc=\t'+str(test_acc1_avg)
                save_to_file(sentence, name)
            #except:
                #print('not success at ratio: ',ratio_number)

if __name__ == '__main__':
  main()
