from typing import List, Sequence
from .warp import *
from sklearn import metrics
import numpy as np
import data.occupancy_flow_metrics_pb2 as occupancy_flow_metrics_pb2
from sklearn.utils.multiclass import type_of_target

def _compute_occupancy_auc(true_occupancy, pred_occupancy):
    gt = true_occupancy.reshape(-1).cpu().numpy().astype(np.int32)
    # print('1', np.sum(gt == 1))
    # print('0', np.sum(gt == 0))
    pred = pred_occupancy.reshape(-1).cpu().numpy()
    # fpr, tpr, thresholds = metrics.roc_curve(gt, pred)
    # return metrics.auc(fpr, tpr)
    
    precision, recall, _thresholds = metrics.precision_recall_curve(gt, pred)
    area = metrics.auc(recall, precision)
    
    return area
    

def _compute_occupancy_soft_iou(true_occupancy, pred_occupancy):
    gt = true_occupancy.reshape(-1)
    pred = pred_occupancy.reshape(-1)
    
    intersection = torch.mean(torch.mul(gt, pred))
    true_sum = torch.mean(gt)
    pred_sum = torch.mean(pred)
    
    score = intersection / (pred_sum + true_sum - intersection + 1e-20)
    
    # print(score)
    return score.cpu().numpy()

def _compute_flow_epe(true_flow, pred_flow):
    true_flow = true_flow
    pred_flow = pred_flow
    diff = true_flow - pred_flow
    true_flow_dx, true_flow_dy = true_flow[..., 0].unsqueeze(-1), true_flow[..., 1].unsqueeze(-1)
    flow_exists = torch.logical_or(
        torch.not_equal(true_flow_dx, 0.0),
        torch.not_equal(true_flow_dy, 0.0),
    ).float()
    
    diff = diff * flow_exists
    epe = torch.linalg.norm(diff, ord = 2, axis = -1, keepdims = True)
    sum_epe = torch.sum(epe)
    sum_flow_exists = torch.sum(flow_exists)
    mean_epe = torch.div(sum_epe, sum_flow_exists + 1e-20)
    
    return mean_epe.cpu().numpy()

def _mean(tensor_list):
    num_tensors = len(tensor_list)
    sum_tensors = np.sum(tensor_list)
    
    return sum_tensors / num_tensors
    

def compute_occupancy_flow_metrics(args, pred_waypoints, gt):
    metrics_dict = {
      'vehicles_observed_auc': [],
      'vehicles_occluded_auc': [],
      'vehicles_observed_iou': [],
      'vehicles_occluded_iou': [],
      'vehicles_flow_epe': [],
      'vehicles_flow_warped_occupancy_auc': [],
      'vehicles_flow_warped_occupancy_iou': [],
    }
    
    batch_size = gt['observed'].shape[0]
        
    pred_waypoints = pred_waypoints.reshape(batch_size, args.grid_height_cells, args.grid_width_cells, -1, args.num_waypoints).contiguous()
    
    # np.save('/GPFS/rhome/ziwang/projects/occupancy_flow/vis/vis_files/metrics_observed.npy', gt['observed'])
    
    gt_observed = gt['observed'].cuda()
    gt_occluded = gt['occluded'].cuda()
    gt_flow = gt['flow'].cuda()
    gt_flow_origin_occupancy = gt['flow_origin_occupancy'].cuda()
    
    warped_flow_origins = []
    for k in range(args.num_waypoints):
        warped_flow_origins.append(get_warped_occupancy_pytorch(gt_flow_origin_occupancy[..., k], pred_waypoints[:, :, :, 2:, k]))
        
    
    for k in range(args.num_waypoints):
        true_observed_occupancy = gt_observed[..., k]
        pred_observed_occupancy = torch.sigmoid(pred_waypoints[:, :, :, :1, k])
        true_occluded_occupancy = gt_occluded[..., k]
        pred_occluded_occupancy = torch.sigmoid(pred_waypoints[:, :, :, 1:2, k])
        true_flow = gt_flow[..., k]
        pred_flow = pred_waypoints[:, :, :, 2:, k]
        
        metrics_dict['vehicles_observed_auc'].append(
            _compute_occupancy_auc(true_observed_occupancy,
                               pred_observed_occupancy))
        metrics_dict['vehicles_occluded_auc'].append(
            _compute_occupancy_auc(true_occluded_occupancy,
                                pred_occluded_occupancy))
        metrics_dict['vehicles_observed_iou'].append(
            _compute_occupancy_soft_iou(true_observed_occupancy,
                                        pred_observed_occupancy))
        metrics_dict['vehicles_occluded_iou'].append(
            _compute_occupancy_soft_iou(true_occluded_occupancy,
                                        pred_occluded_occupancy))
        
        metrics_dict['vehicles_flow_epe'].append(
            _compute_flow_epe(true_flow, pred_flow))
        
        true_all_occupancy = torch.clamp(
            true_observed_occupancy + true_occluded_occupancy, 0, 1)
        
        pred_all_occupancy = torch.clamp(
            pred_observed_occupancy + pred_occluded_occupancy, 0, 1)
        
        flow_warped_origin_occupancy = warped_flow_origins[k]
        # print(torch.isnan(flow_warped_origin_occupancy).int().sum())
        flow_grounded_pred_all_occupancy = (torch.mul(pred_all_occupancy, flow_warped_origin_occupancy))
        
        # print(torch.isnan(flow_grounded_pred_all_occupancy).int().sum())
        metrics_dict['vehicles_flow_warped_occupancy_auc'].append(
            _compute_occupancy_auc(true_all_occupancy,
                                   flow_grounded_pred_all_occupancy))
        metrics_dict['vehicles_flow_warped_occupancy_iou'].append(
            _compute_occupancy_soft_iou(true_all_occupancy,
                                        flow_grounded_pred_all_occupancy))
        
    metrics = occupancy_flow_metrics_pb2.OccupancyFlowMetrics()
    
    metrics.vehicles_observed_auc = _mean(metrics_dict['vehicles_observed_auc'])
    metrics.vehicles_occluded_auc = _mean(metrics_dict['vehicles_occluded_auc'])
    metrics.vehicles_observed_iou = _mean(metrics_dict['vehicles_observed_iou'])
    metrics.vehicles_occluded_iou = _mean(metrics_dict['vehicles_occluded_iou'])
    metrics.vehicles_flow_epe = _mean(metrics_dict['vehicles_flow_epe'])
    metrics.vehicles_flow_warped_occupancy_auc = _mean(metrics_dict['vehicles_flow_warped_occupancy_auc'])
    metrics.vehicles_flow_warped_occupancy_iou = _mean(metrics_dict['vehicles_flow_warped_occupancy_iou'])
        
    return metrics