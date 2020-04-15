import torch

def euclidean_dist(x, y, transform=True):
  bs = x.shape[0]
  if transform:
    num_proto = y.shape[0]
    query_lst = []
    for i in range(bs):
      ext_query = x[i, :].repeat(num_proto, 1)
      query_lst.append(ext_query)
    x = torch.cat(query_lst, dim=0)
    y = y.repeat(bs, 1)
 
  return torch.pow(x - y, 2).sum(-1)
