from collections import OrderedDict
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
from .projective_ops import projective_transform


def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err


def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s

def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:,jj] * Ps[:,ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()

        if do_scale:
            s = fit_scale(dP, dG)
            dG = dG.scale(s[:,None])
        
        # pose error
        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3,3], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean())

        elif isinstance(dG, Sim3):
            tau, phi, sig = d.split([3,3,1], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean() + 
                0.05 * sig.norm(dim=-1).mean())
            
        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'rot_error': r_err.mean().item(),
        'tr_error': t_err.mean().item(),
        'bad_rot': (r_err < .1).float().mean().item(),
        'bad_tr': (t_err < .01).float().mean().item(),
    }

    return geodesic_loss, metrics


def residual_loss(residuals, gamma=0.9):
    """ loss on system residuals """
    residual_loss = 0.0
    n = len(residuals)

    for i in range(n):
        w = gamma ** (n - i - 1)
        residual_loss += w * residuals[i].abs().mean()

    return residual_loss, {'residual': residual_loss.item()}


def flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph, gamma=0.9):
    """ optical flow loss """

    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j)==1]

    ii, jj, kk = graph_to_edge_list(graph)
    coords0, val0 = projective_transform(Ps, disps, intrinsics, ii, jj)
    val0 = val0 * (disps[:,ii] > 0).float().unsqueeze(dim=-1)

    n = len(poses_est)
    flow_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        coords1, val1 = projective_transform(poses_est[i], disps_est[i], intrinsics, ii, jj)

        v = (val0 * val1).squeeze(dim=-1)
        epe = v * (coords1 - coords0).norm(dim=-1)
        flow_loss += w * epe.mean()

    epe = epe.reshape(-1)[v.reshape(-1) > 0.5]
    metrics = {
        'f_error': epe.mean().item(),
        '1px': (epe<1.0).float().mean().item(),
    }

    return flow_loss, metrics



####################################


def artifical_mask_loss(Ps, poses_est, disps_est, agg_flow, logits, intrinsics, graph, gamma=0.9):
    
    
    art_mask_losses = 0.0
    n = len(poses_est)
        
    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j)==1]

    ii, jj, kk = graph_to_edge_list(graph)
    
    ### target
    ht, wd = disps_est[i][:,ii].shape[2:]
    yi, xi = torch.meshgrid(
        torch.arange(ht).to(disps_est[i][:,ii].device).float(),
        torch.arange(wd).to(disps_est[i][:,ii].device).float())
    

    for i in range(n):
        artifical_mask = 0
        proj, val = projective_transform(poses_est[i], disps_est[i], intrinsics, ii, jj)
        # coords, val is tensors
        flow = [xi, yi] + agg_flow
        
        abs_diff = torch.abs(proj - flow)
        l2_loss = (abs_diff**2).mean(1, True)
        if l2_loss > 0.5: artifical_mask = 1 
        
        
        
        art_mask_losses += (artifical_mask*math.log(logits)+(1-artifical_mask)*math.log(1-logits))

    return art_mask_losses / n
    
    


def flow_photometric_loss(Ps, poses_est, disps_est, agg_flow, intrinsics, graph, gamma=0.9):
    
    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j)==1]

    ii, jj, kk = graph_to_edge_list(graph)
    
    n = len(poses_est)
    flow_loss = 0.0
    
    
    
    ht, wd = disps_est[i][:,jj].shape[2:]
    yj, xj = torch.meshgrid(
        torch.arange(ht).to(disps_est[i][:,jj].device).float(),
        torch.arange(wd).to(disps_est[i][:,jj].device).float())
    
    ht, wd = disps_est[i][:,ii].shape[2:]
    yi, xi = torch.meshgrid(
        torch.arange(ht).to(disps_est[i][:,ii].device).float(),
        torch.arange(wd).to(disps_est[i][:,ii].device).float())
    
    for i in range(n):
        
        pred = [xi, yi] + agg_flow
        abs_diff = torch.abs([xj, yj] - pred)
        l1_loss = abs_diff.mean(1, True)
        
        flow_loss += l1_loss


def compute_reprojection_loss(self, pred, target, logits):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = self.ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    
    reprojection_loss *= logits

    return reprojection_loss



def geometry_photometric_loss(Ps, poses_est, disps_est, logits, intrinsics, graph, gamma=0.9):
    
    reprojection_losses = 0.0
    n = len(poses_est)
        
    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j)==1]

    ii, jj, kk = graph_to_edge_list(graph)
    
    ### target
    ht, wd = disps_est[i][:,ii].shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps_est[i][:,ii].device).float(),
        torch.arange(wd).to(disps_est[i][:,ii].device).float())
    

    for i in range(n):
        pred, val = projective_transform(poses_est[i], disps_est[i], intrinsics, ii, jj)
        # coords, val is tensors
        reprojection_losses += compute_reprojection_loss([x, y], pred, logits)

        
    return reprojection_losses/n

def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = SSIM(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


#### from monodepth2 code

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        # 입력 경계의 반사를 사용하여 상/하/좌/우에 입력 텐서를 추가로 채웁니다.
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # 
        # shape : (xh, xw) -> (xh + 2, xw + 2)
        x = self.refl(x) 
        # shape : (yh, yw) -> (yh + 2, yw + 2)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)