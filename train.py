import torch
import numpy as np
from config import parse_args
from data.dataset_v2 import *
from torch.utils.data import DataLoader
from utils.losses import *
import os
from models.occupancyflow_field import *
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

def train(args):

    # model = OccupancyFlow_Field(args).cuda()

    # collate_fn = collater_v2(args)
    # train_dataset = OccupancyFlow_Dataset(preprocessed_path = args.train_preprocessed_file_path)
    # train_loader = DataLoader(dataset = train_dataset,batch_size = args.batch_size, collate_fn = collate_fn, shuffle = False)
    
    # loss_fn = Loss(args)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    collate_fn = collater_v2(args)
    train_dataset = OccupancyFlow_Dataset(args, mode = 'train')
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    
    val_dataset = OccupancyFlow_Dataset(args, mode = 'val')
    val_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    
    loss_fn = Loss(args)
    model = OccupancyFlow_Field(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch_num):
        model.train()
        loss_list = []
        observed_loss_list = []
        occluded_loss_list = []
        flow_loss_list = []
        trace_loss_list = []
        batch_idx = 0
        for agent_points_features, rg_points_features, gt in tqdm(train_loader):
            
            pred_result = model(agent_points_features, rg_points_features)
            # print(pred_result)
            optimizer.zero_grad() 
            observed_loss, occluded_loss, flow_loss = loss_fn(pred_result, gt)
            loss = observed_loss + occluded_loss + flow_loss# + trace_loss
            observed_loss_list.append(observed_loss.item())
            occluded_loss_list.append(occluded_loss.item())
            flow_loss_list.append(flow_loss.item())
            # trace_loss_list.append(trace_loss)
            loss.backward()  
            optimizer.step()   
            loss_list.append(loss.item())
        # if args.use_trace_loss:
        #     print("training: epoch={}, loss={}, observed={}, occluded={}, flow={}, trace={}".format(epoch, \
        #         torch.sum(torch.tensor(loss_list)), \
        #         torch.sum(torch.tensor(observed_loss_list)), \
        #         torch.sum(torch.tensor(occluded_loss_list)), \
        #         torch.sum(torch.tensor(flow_loss_list)), \
        #         torch.sum(torch.tensor(trace_loss_list)), \
        #         ))
        # else:
            if batch_idx % args.print_freq == 0:
                print("training: epoch={}, loss={}, observed={}, occluded={}, flow={}".format(epoch, \
                    torch.sum(torch.tensor(loss_list)), \
                    torch.sum(torch.tensor(observed_loss_list)), \
                    torch.sum(torch.tensor(occluded_loss_list)), \
                    torch.sum(torch.tensor(flow_loss_list)), \
                    ))
                loss_list = []
                observed_loss_list = []
                occluded_loss_list = []
                flow_loss_list = []
                
            batch_idx += 1

        for agent_points_features, rg_points_features, gt in tqdm(val_loader):
            batch_size = gt['observed'].shape[0]
        
        

if __name__ == '__main__':
    args = parse_args()
    train(args)