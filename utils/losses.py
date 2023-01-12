import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .warp import *
  
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.occupancy_loss_fn = nn.BCEWithLogitsLoss()
      
    def occupancy_loss(self, pred_occupancy, true_occupancy):
        return self.occupancy_loss_fn(pred_occupancy, true_occupancy)
      
    def flow_loss(self, pred_flow, true_flow):
        diff = true_flow - pred_flow
        # print('diff', torch.isnan(diff).int().sum())
        true_flow_dx, true_flow_dy = true_flow[..., 0].unsqueeze(-1), true_flow[..., 1].unsqueeze(-1)
        flow_exists = torch.logical_or(
          torch.not_equal(true_flow_dx, 0.0),
          torch.not_equal(true_flow_dy, 0.0),
        ).float()
        diff = diff * flow_exists
        # print('diff 2', torch.isnan(diff).int().sum())
        diff_norm = torch.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
        # print('diff_norm', torch.isnan(diff_norm).int().sum())
        # print(torch.sum(flow_exists))
        mean_diff = torch.div(torch.sum(diff_norm), torch.sum(flow_exists) / 2 + 1e-20)
        # print('mean_diff', torch.isnan(mean_diff).int().sum())
        return mean_diff
      
    def forward(self, pred_result, gt):
        true_observed_occupancy = gt['observed'].cuda()
        true_occluded_occupancy = gt['occluded'].cuda()
        true_flow = gt['flow'].cuda()
        flow_origin_occupancy = gt['flow_origin_occupancy'].cuda()
        batch_size = true_observed_occupancy.shape[0]
        
        pred_result = pred_result.reshape(batch_size, self.args.grid_height_cells, self.args.grid_width_cells, -1, self.args.num_waypoints).contiguous()
        
        pred_observed_occupancy = pred_result[:, :, :, :1, :]
        observed_loss = self.occupancy_loss(pred_observed_occupancy, true_observed_occupancy)
        
        pred_occluded_occupancy = pred_result[:, :, :, 1:2, :]
        occluded_loss = 0
        if self.args.use_occluded_loss:
            occluded_loss = self.occupancy_loss(pred_occluded_occupancy, true_occluded_occupancy)
            
        
        pred_flow = pred_result[:, :, :, 2:, :]
        
        flow_loss = 0
        trace_loss = 0
        # pred_all_occupancy = pred_observed_occupancy + pred_occluded_occupancy
        # pred_all_occupancy = pred_all_occupancy / pred_all_occupancy.max()
        # pred_all_occupancy = torch.clamp(pred_observed_occupancy + pred_occluded_occupancy, min = 0.0, max = 1.0)
        pred_all_occupancy = pred_observed_occupancy + pred_occluded_occupancy
        true_all_occupancy = torch.clamp(true_observed_occupancy + true_occluded_occupancy, min = 0.0, max = 1.0)
        for k in range(self.args.num_waypoints):
            # print(k, torch.isnan(pred_flow[..., k]).int().sum(), torch.isnan(true_flow[..., k]).int().sum())
            if self.args.use_flow_loss:
                flow_loss += self.flow_loss(pred_flow[..., k], true_flow[..., k])
            if self.args.use_trace_loss:
                warped_occupancy = get_warped_occupancy_pytorch(flow_origin_occupancy[..., k], pred_flow[..., k])
                trace_loss += self.occupancy_loss(torch.mul(warped_occupancy, pred_all_occupancy[..., k]), true_all_occupancy[..., k])
        
        return self.args.occupancy_weight * observed_loss, \
                    self.args.occupancy_weight * occluded_loss, \
                    self.args.flow_weight * flow_loss, \
                    self.args.trace_weight * trace_loss
        
        
     
# if __name__ == '__main__':
#     occupancy = torch.rand(2, 256, 256, 1, 8)
#     flow = torch.rand(2, 256, 256, 2, 8)
#     flow_origin_occupancy = torch.rand(2, 256, 256, 1, 8)
#     print(flow_loss(flow[..., 0]+1, flow[..., 0], 1))
#     print(trace_loss(occupancy, flow, flow_origin_occupancy))