import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        
        self.logits_encoder = nn.Sequential(
            nn.Conv2d(1, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())
        
        
        self.classification_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, 3, padding=1),
        )

        self.gru = ConvGRU(128, 128+128+80)
        self.agg = GraphAgg()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=3, stride=1, padding=1)

    def forward(self, net, inp, corr, logits, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        # print("flow check2:", flow.size())
        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)
        # if logits is not None:
        #     logits = logits.to(net.device)

        # print("!!corr 크기:", corr.size())
        # print("!!flow 크기:", flow.size())
        # print("!!logits 크기:", logits.size())

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        logits = logits.view(batch*num, -1, ht, wd)
        # print("Flow dimensions before reshaping:", flow.size())

        flow = flow.view(batch*num, -1, ht, wd)

        # batch_size = flow.size(0)
        # logits = logits.repeat(batch_size, 1, 1, 1)  


        # print("@corr 크기:", corr.size())
        # print("@flow 크기:", flow.size())
        # print("@logits 크기:", logits.size())

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        logits = self.logits_encoder(logits)

        # print("corr 크기:", corr.size())
        # print("flow 크기:", flow.size())
        # print("logits 크기:", logits.size())
        

        
        concat_vals = [corr, flow, logits]
        
        cor_flo_logits = torch.cat(concat_vals, dim=1)
        out = self.relu(self.conv(cor_flo_logits))
        motion_features =  torch.cat((out, logits, flow), dim=1)
        # print("flow&&&", flow.size())
        # print("logits&&&", logits.size())
        
        inp = torch.cat((inp, motion_features), dim=1)
        # print("Before gru:", net.shape)
        
        net = self.gru(net, inp)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)
        delta_logits = self.classification_head(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()
        delta_logits = delta_logits.permute(0,1,3,4,2)[...,:1].contiguous()
        # print("output delta_logits : ", delta_logits.size())
        # print("output delta : ", delta.size())

        net = net.view(*output_dim)        
        

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, delta_logits, eta, upmask

        else:
            return net, delta, weight, delta_logits


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()


    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        b = images.shape[0]
        # print("b", b)

        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        # print("coords0 check:", coords0.size())
        
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()
        
        logits = torch.zeros((b, coords1.size(1), ht//8, wd//8, 1), dtype=torch.float32, device=images.device)

        # print("coords1 check:", coords1.size())
        # print("logits check:", logits.size())

        Gs_list, disp_list, residual_list, logits_list = [], [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()
            logits = logits.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0
            # print("flow check:", flow.size())
            # print("resd check:", resd.size())

            motion = torch.cat([flow, resd], dim=-1)
            # print("motion1 check:", motion.size())
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)
            # print("motion2 check:", motion.size())

            logits_inp = logits.permute(0,1,4,2,3).clamp(0.0, 1.0)


            # print("logits check:", logits.size())
            net, delta, weight, delta_logits, eta, upmask = \
                self.update(net, inp, corr, logits_inp, motion, ii, jj)
            
            # print("delta check:", delta.size())

            target = coords1 + delta


            # print("delta_logits check:", delta_logits.size())
            # print("logits check:", logits.size())

            logits = logits + delta_logits

            # print("delta_logits check:", delta_logits.size())
            

            for i in range(2):
                Gs, disps = BA(target, weight, logits, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)
            logits_list.append(logits)


        return Gs_list, disp_list, residual_list