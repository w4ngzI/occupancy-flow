import argparse
import os

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch_num', type = int, default = 100)
    parse.add_argument('--learning_rate', type = float, default = 1e-3)
    parse.add_argument('--batch_size', type = int, default = 2)
    parse.add_argument('--train_preprocessed_file_path', type = str, default = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/train_preprocessed_data')
    parse.add_argument('--val_preprocessed_file_path', type = str, default = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/val_preprocessed_data')
    parse.add_argument('--exp_name', type = str, default = 'test')
    parse.add_argument('--points_per_side_length', type = int, default = 48)
    parse.add_argument('--points_per_side_width', type = int, default = 16)
    parse.add_argument('--grid_height_cells', type = int, default = 256)
    parse.add_argument('--grid_width_cells', type = int, default = 256)
    parse.add_argument('--pixels_per_meter', type = float, default = 3.2)
    parse.add_argument('--sdc_x_in_grid', type = int, default = 128)
    parse.add_argument('--sdc_y_in_grid', type = int, default = 192)
    parse.add_argument('--type_num', type = int, default = 3)
    parse.add_argument('--time_num', type = int, default = 11)
    parse.add_argument('--embedding_dim_type', type = int, default = 16)
    parse.add_argument('--embedding_dim_time', type = int, default = 16)
    parse.add_argument('--max_num_points', type = int, default = 32)
    parse.add_argument('--num_waypoints', type = int, default = 8)
    parse.add_argument('--num_pred_channels', type = int, default = 4)
    parse.add_argument('--occupancy_weight', type = float, default = 1)
    parse.add_argument('--flow_weight', type = float, default = 0.001)
    parse.add_argument('--trace_weight', type = float, default = 0.1)
    parse.add_argument('--compound_coef', type = int, default = 5)
    parse.add_argument('--pillar_encoding_dim_agent', type = int, default = 64)   
    parse.add_argument('--pillar_encoding_dim_rg', type = int, default = 64) 
    parse.add_argument('--pred_channels_num', type = int, default = 32)
    parse.add_argument('--use_occluded_loss', action = 'store_true', default = False)
    parse.add_argument('--use_flow_loss', action = 'store_true', default = False)
    parse.add_argument('--use_trace_loss', action = 'store_true', default = False)
    parse.add_argument("--local_rank", default = os.getenv('LOCAL_RANK', -1), type = int)
    parse.add_argument('--num_workers', type = int, default = 0)
    parse.add_argument('--use_zero', action = 'store_true', default = True)
    parse.add_argument('--print_freq', type = int, default = 100)
    parse.add_argument('--resume_training', action = 'store_true', default = False)
    parse.add_argument('--checkpoint_path', type = str, default = '.checkpoints/test/epoch_0.pth')
    parse.add_argument('--valid_val_ids_path', type = str, default = './valid_val_ids.pkl')
    parse.add_argument('--train_mode', type = str, default = 'train')
    parse.add_argument('--use_20data', action = 'store_true', default = False)
    
    args = parse.parse_args()
    return args