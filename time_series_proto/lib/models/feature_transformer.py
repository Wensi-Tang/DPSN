import torch
import torch.nn as nn
from .initialization import init_kaiming

class linear_transform(nn.Module):
  def __init__(self, fea_dim, hidden_dim=2048, out_dim=2048):
    super(linear_transform, self).__init__()
    self.fc1 = nn.Linear(fea_dim, hidden_dim)
    self.dp = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(hidden_dim, out_dim)
    self.apply(weights_init)

  def forward(self,x):
    out = self.fc1(x)
    out = self.dp(out)
    out = self.fc2(out)
    return out

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      m.weight.data.normal_(0, math.sqrt(2. / n))
      if m.bias is not None:
          m.bias.data.zero_()
  elif classname.find('BatchNorm') != -1:
      m.weight.data.fill_(1)
      m.bias.data.zero_()
  elif classname.find('Linear') != -1:
      n = m.weight.size(1)
      m.weight.data.normal_(0, 0.01)
      m.bias.data = torch.ones(m.bias.data.size())
   


