import torch
import numpy as np
from config import parse_args
from data.dataset_v2 import *
from torch.utils.data import DataLoader
from utils.losses import *
import os
from torch.utils.data.distributed import DistributedSampler
from models.occupancyflow_field import *
from tqdm import tqdm
from utils.calculate_metrics import *
from torch.distributed.optim import ZeroRedundancyOptimizer
import logging
import pathlib
import time

device = "cuda:0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def save_model(args, model, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    pathlib.Path('./checkpoints/{}'.format(args.exp_name)).mkdir(exist_ok=True) 
    model_checkpoint = os.path.join('./checkpoints', args.exp_name, "epoch_{}.pth".format(epoch))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join('./checkpoints/{}'.format(args.exp_name)))


def train(args):
    num_gpus = torch.cuda.device_count()
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        model = OccupancyFlow_Field_v3(args).to(device)
    else:
        model = OccupancyFlow_Field_v3(args).cuda()
        
    if args.resume_training:
        print('checkpoint path', args.checkpoint_path)
        model.load_state_dict(torch.load(args.checkpoint_path))
        print("loaded model")
    
    if num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)
    
    
    collate_fn = collater_v2(args)
    train_dataset = OccupancyFlow_Dataset(args, mode = args.train_mode)
    val_dataset = OccupancyFlow_Dataset(args, mode = 'val')
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(dataset = train_dataset, sampler = train_sampler, 
                                batch_size = args.batch_size, collate_fn = collate_fn,
                                shuffle = False, num_workers = args.num_workers, 
                                pin_memory = True, prefetch_factor = 2)
        
        # val_sampler = DistributedSampler(val_dataset)
        # val_loader = DataLoader(dataset = val_dataset, sampler = val_sampler, 
        #                         batch_size = args.batch_size, collate_fn = collate_fn,
        #                         shuffle = False, num_workers = args.num_workers, 
        #                         pin_memory = True, prefetch_factor = 2)
        val_loader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn, num_workers = args.num_workers, prefetch_factor = 2)
    else:
        train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
        val_loader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
        
    loss_fn = Loss(args)
    
    if args.use_zero and args.local_rank != -1:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_params = count_parameters(model)
    if args.local_rank in [-1, 0]:
        logger.info("Training parameters %s", args)
        logger.info("Total Parameter: \t%2.1fM" % num_params)
    
    for epoch in range(args.epoch_num):
        if args.local_rank in [-1, 0]:
            logger.info("\n********** Running Training On Epoch {} **********".format(epoch))
        total_loss_am = AverageMeter()
        observed_loss_am = AverageMeter()
        occluded_loss_am = AverageMeter()
        flow_loss_am = AverageMeter()
        trace_loss_am = AverageMeter()
        load_data_time_am = AverageMeter()
        
        model.train()
        # if args.local_rank != -1:
        #     train_sampler.set_epoch(epoch)
        batch_idx = 0
        start_load_data = time.perf_counter()
        for agent_points_features, rg_points_features, gt in tqdm(train_loader):
            batch_size = gt['observed'].shape[0]
            end_load_data = time.perf_counter()
            load_data_time_am.update(end_load_data - start_load_data)
            pred_result = model(agent_points_features, rg_points_features)
            optimizer.zero_grad() 
            observed_loss, occluded_loss, flow_loss, trace_loss = loss_fn(pred_result, gt)
            total_loss = observed_loss + occluded_loss + flow_loss + trace_loss
            # total_loss = occluded_loss + flow_loss + trace_loss
                
            total_loss_am.update(total_loss.item(), batch_size)
            observed_loss_am.update(observed_loss.item(), batch_size)
            if args.use_occluded_loss:
                occluded_loss_am.update(occluded_loss.item(), batch_size)
            if args.use_flow_loss:
                flow_loss_am.update(flow_loss.item(), batch_size)
            if args.use_trace_loss:
                trace_loss_am.update(trace_loss.item(), batch_size)
            
            total_loss.backward()  
            optimizer.step()   
            
            if batch_idx % args.print_freq == 0:
                print("training: epoch={}, batch={}, loss={}, observed={}, occluded={}, flow={}, trace={}".format(epoch, batch_idx,\
                        total_loss_am.avg, observed_loss_am.avg, occluded_loss_am.avg, flow_loss_am.avg, trace_loss_am.avg))
                if args.local_rank in [-1, 0]:
                    logger.info("training: epoch={}, batch={}, data time={}, loss={}, observed={}, occluded={}, flow={}, trace={}".format(epoch, batch_idx, load_data_time_am.avg, total_loss_am.avg, observed_loss_am.avg, occluded_loss_am.avg, flow_loss_am.avg, trace_loss_am.avg))
                
                total_loss_am.reset()
                observed_loss_am.reset()
                occluded_loss_am.reset()
                flow_loss_am.reset()
                trace_loss_am.reset()
                load_data_time_am.reset()
                
            batch_idx += 1
            start_load_data = time.perf_counter()
        if args.local_rank != -1:
            torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            logger.info("\n********** Running Validation On Epoch {} **********".format(epoch))
            logger.info("Total num steps = %d", len(val_loader))
        model.eval()
        vehicles_observed_auc = AverageMeter()
        vehicles_occluded_auc = AverageMeter()
        vehicles_observed_iou = AverageMeter()
        vehicles_occluded_iou = AverageMeter()
        vehicles_flow_epe = AverageMeter()
        vehicles_flow_warped_occupancy_auc = AverageMeter()
        vehicles_flow_warped_occupancy_iou = AverageMeter()
        for agent_points_features, rg_points_features, gt in tqdm(val_loader):
            batch_size = gt['observed'].shape[0]
            with torch.no_grad():
                pred_result = model(agent_points_features, rg_points_features)
                metrics = compute_occupancy_flow_metrics(args, pred_result, gt)
                vehicles_observed_auc.update(metrics.vehicles_observed_auc, batch_size)
                vehicles_occluded_auc.update(metrics.vehicles_occluded_auc, batch_size)
                vehicles_observed_iou.update(metrics.vehicles_observed_iou, batch_size)
                vehicles_occluded_iou.update(metrics.vehicles_occluded_iou, batch_size)
                vehicles_flow_epe.update(metrics.vehicles_flow_epe, batch_size)
                vehicles_flow_warped_occupancy_auc.update(metrics.vehicles_flow_warped_occupancy_auc, batch_size)
                vehicles_flow_warped_occupancy_iou.update(metrics.vehicles_flow_warped_occupancy_iou, batch_size)
        
        if args.local_rank in [-1, 0]:
            logger.info("vehicles_observed_auc: {}".format(vehicles_observed_auc.val))
            logger.info("vehicles_occluded_auc: {}".format(vehicles_occluded_auc.val))
            logger.info("vehicles_observed_iou: {}".format(vehicles_observed_iou.val))
            logger.info("vehicles_occluded_iou: {}".format(vehicles_occluded_iou.val))
            logger.info("vehicles_flow_epe: {}".format(vehicles_flow_epe.val))
            logger.info("vehicles_flow_warped_occupancy_auc: {}".format(vehicles_flow_warped_occupancy_auc.val))
            logger.info("vehicles_flow_warped_occupancy_iou: {}".format(vehicles_flow_warped_occupancy_iou.val))

            # save_model(args, model, epoch)

if __name__ == '__main__':
    args = parse_args()
    pathlib.Path('./logs').mkdir(exist_ok=True) 
        
    logging.basicConfig(level=logging.DEBUG,
                        filename = 'logs/' + args.exp_name + '.log',
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        )
    
    train(args)