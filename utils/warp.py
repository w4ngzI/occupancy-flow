import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_warped_occupancy(occupancy, flow):
        flow_indicator = torch.logical_or(flow[..., 0] != 0, flow[..., 1] != 0) 
        indices_no_flow = torch.nonzero(~flow_indicator)
        indices_flow = torch.nonzero(flow_indicator) 
           
        output = torch.zeros(occupancy.shape).cuda()
        output[indices_no_flow[:, 0], indices_no_flow[:, 1], indices_no_flow[:, 2]] = occupancy[indices_no_flow[:, 0], indices_no_flow[:, 1], indices_no_flow[:, 2]]
        batch_indices = indices_flow[:, 0]
        flow_y = flow[indices_flow[:, 0], indices_flow[:, 1], indices_flow[:, 2], 1]
        flow_x = flow[indices_flow[:, 0], indices_flow[:, 1], indices_flow[:, 2], 0]
        
        y_indices = torch.clamp((indices_flow[:, 1] + flow_y + 0.5), 0, 255).long()
        x_indices = torch.clamp((indices_flow[:, 2] + flow_x + 0.5), 0, 255).long()
        output[batch_indices, y_indices, x_indices] = occupancy[indices_flow[:, 0], indices_flow[:, 1], indices_flow[:, 2]]     
        return output

def get_warped_occupancy_pytorch(occupancy, flow):
        B, H, W, C = occupancy.shape
        h = torch.arange(0, H).view(1, -1).repeat(H, 1)
        w = torch.arange(0, W).view(-1, 1).repeat(1, W)
        h = h.view(1, 1, H, W).repeat(B, 1, 1, 1)
        w = w.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((h, w), 1).float().permute(0, 2, 3, 1).cuda().double()
        
        warped_indices = grid
        warped_indices[..., 0] = grid[..., 0] + flow[..., 0]   
        warped_indices[..., 1] = grid[..., 1] + flow[..., 1]
        warped_indices[:, :, :, 0] = 2.0 * warped_indices[:, :, :, 0].clone() / (H - 1) - 1.0
        warped_indices[:, :, :, 1] = 2.0 * warped_indices[:, :, :, 1].clone() / (W - 1) - 1.0
        
        output = F.grid_sample(occupancy.permute(0, 3, 1, 2), warped_indices, padding_mode = 'zeros', align_corners = False)
        
        return output.permute(0, 2, 3, 1)