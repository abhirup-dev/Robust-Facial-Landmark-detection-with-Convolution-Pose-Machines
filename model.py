import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ITN_CPM(nn.Module):
    def __init__(self, params):
        super(ITN_CPM, self).__init__()
        self.features = nn.Sequential(
              nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

    #     self.downsample = 8
        self.params = params
        pts_num = params['num_pts']+1
        self.CPM_features = nn.Sequential(
              nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
              nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2
        self.stage1 = nn.Sequential(
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
              nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        self.stage2 = nn.Sequential(
              nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
              nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
        self.stage3 = nn.Sequential(
              nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
              nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
    
    def specify_parameter(self, base_lr, base_weight_decay):
        params_dict = [ 
            {'params': get_parameters(self.features,   bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.features,   bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
            {'params': get_parameters(self.CPM_features, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.CPM_features, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
            {'params': get_parameters(self.stage1,      bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.stage1,      bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
            {'params': get_parameters(self.stage2,      bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.stage2,      bias=True ), 'lr': base_lr*8, 'weight_decay': 0},
            {'params': get_parameters(self.stage3,      bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay},
            {'params': get_parameters(self.stage3,      bias=True ), 'lr': base_lr*8, 'weight_decay': 0}
        ]
        return params_dict
    
    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size = inputs.size(0)
        num_pts = self.params['num_pts']

        batch_cpms, batch_locs, batch_scos = [], [], []     # [Squence, Points]
        
        feature  = self.features(inputs)
        feature = self.CPM_features(feature)
        stage1 = self.stage1(feature)
        cpm_stage2 = self.stage2(torch.cat([feature, stage1], 1))
        cpm_stage3 = self.stage3(torch.cat([feature, cpm_stage2], 1))
        batch_cpms = [stage1, cpm_stage2, cpm_stage3]
#         print(feature.shape)
#         print(stage1.shape)
#         print(cpm_stage2.shape)
#         print(cpm_stage3.shape)
#         print(len(batch_cpms))

        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(cpm_stage3[ibatch], 
                                            self.params['argmax_radius'], self.params['downsample'])
#             print(batch_location.shape, batch_score.shape)
            batch_locs.append( batch_location )
            batch_scos.append( batch_score )
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

        return batch_cpms, batch_locs, batch_scos
    
def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0, 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()
  
    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
    affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
    affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
    affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    theta = np2variable(affine_parameter, heatmap.is_cuda, False)
    grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = np2variable(torch.arange(-radius, radius+1, out=torch.FloatTensor()), heatmap.is_cuda, False).view(1, 1, radius*2+1)
    Y = np2variable(torch.arange(-radius, radius+1, out=torch.FloatTensor()), heatmap.is_cuda, False).view(1, radius*2+1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
    x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y],1), score

def get_parameters(model, bias):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if bias: 
                yield m.bias
            else: 
                yield m.weight
        elif isinstance(m, nn.BatchNorm2d):
            if bias: 
                yield m.bias
            else:
                yield m.weight

def np2variable(x, is_cuda=True, requires_grad=True, dtype=torch.FloatTensor):
    if isinstance(x, np.ndarray):
        v = torch.autograd.Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    elif isinstance(x, torch.FloatTensor):
        v = torch.autograd.Variable(x.type(dtype), requires_grad=requires_grad)
    else:
        raise Exception('Do not know this type : {}'.format( type(x) ))

    if is_cuda: return v.cuda()
    else: return v
    
def variable2np(x):
    if x.is_cuda:
        x = x.cpu()
    if isinstance(x, torch.autograd.Variable):
        return x.data.numpy()
    else:
        return x.numpy()