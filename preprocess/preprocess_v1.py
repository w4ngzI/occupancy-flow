from preprocess_tools import *
import pathlib
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib

# from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# import tensorflow_graphics.image.transformer as tfg_transformer

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
import pickle 
from tqdm import tqdm

# DATASET_FOLDER = '/nvme/jxs/WaymoOpenMotion/scenario'
DATASET_FOLDER = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset'

# TFRecord dataset.
TRAIN_FILES = f'{DATASET_FOLDER}/training/training.tfrecord*'
# TRAIN_FILES = 'training.tfrecord*'
VAL_FILES = f'{DATASET_FOLDER}/validation/validation.tfrecord*'
TEST_FILES = f'{DATASET_FOLDER}/testing/testing.tfrecord*'


# Text files containing validation and test scenario IDs for this challenge.
VAL_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
TEST_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

config_text = """
num_past_steps: 10
num_future_steps: 80
num_waypoints: 8
cumulative_waypoints: false
normalize_sdc_yaw: true
grid_height_cells: 256
grid_width_cells: 256
sdc_y_in_grid: 192
sdc_x_in_grid: 128
pixels_per_meter: 3.2
agent_points_per_side_length: 48
agent_points_per_side_width: 16
"""
raw_dataset = tf.data.TFRecordDataset(['/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/training/training.tfrecord-00000-of-01000'])


# filenames = tf.io.matching_files(VAL_FILES)
# raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='')
# for raw_record_index, raw_record in tqdm(enumerate(raw_dataset)):
#     # print(raw_record_index)
#     proto_string = raw_record.numpy()
#     proto = scenario_pb2.Scenario()
#     proto.ParseFromString(proto_string)
    
#     inputs = read_data_proto(proto)
#     # inputs_numpy = {k:np.array(v, dtype=object) for k, v in inputs.items()}
#     config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
#     text_format.Parse(config_text, config)
    
#     inputs_batch = {}
#     for k, v in inputs.items():
#         if k == 'scenario/id':
#             inputs_batch[k] = v
#             continue
#         if k[:7] == 'traffic':
#             inputs_batch[k] = v
#             continue
#         inputs_batch[k] = np.expand_dims(np.array(v, dtype = np.float32), 0)

#     # with open('/nvme/jxs/WaymoOpenMotion/scenario/occupancy_flow/val_preprocessed_data/{}.pkl'.format(raw_record_index), 'wb') as fp:
#     #     pickle.dump(inputs_batch, fp)
        
#     inputs_batch = add_sdc_fields(inputs_batch)
#     inputs_batch = numpy_to_tf(inputs_batch)
    
#     timestep_grids = create_ground_truth_timestep_grids(inputs=inputs_batch, config=config)
#     true_waypoints = create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=config)
    
#     save_file = {}
#     observed = []
#     occluded = []
#     flow = []
#     flow_origin_occupancy = []

#     for i in range(len(true_waypoints.vehicles.observed_occupancy)):
#         observed.append(np.expand_dims(np.array(true_waypoints.vehicles.observed_occupancy[i], dtype = np.float), -1))
#         occluded.append(np.expand_dims(np.array(true_waypoints.vehicles.occluded_occupancy[i], dtype = np.float), -1))
#         flow.append(np.expand_dims(np.array(true_waypoints.vehicles.flow[i], dtype = np.float), -1))
#         flow_origin_occupancy.append(np.expand_dims(np.array(true_waypoints.vehicles.flow_origin_occupancy[i], dtype = np.float), -1))
        
#     save_file['scenario/id'] = inputs_batch['scenario/id']
#     save_file["observed"] = np.concatenate(observed, -1)
#     save_file["occluded"] = np.concatenate(occluded, -1)
#     save_file["flow"] = np.concatenate(flow, -1)
#     save_file["flow_origin_occupancy"] = np.concatenate(flow_origin_occupancy, -1)

#     with open('/nvme/jxs/WaymoOpenMotion/scenario/occupancy_flow/val_gt/{}.pkl'.format(raw_record_index), 'wb') as fp:
#         pickle.dump(save_file, fp)
        
        
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
text_format.Parse(config_text, config)        

# filenames = tf.io.matching_files(TRAIN_FILES)
# raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='')
for raw_record_index, raw_record in tqdm(enumerate(raw_dataset)):
    # print(raw_record_index)
    proto_string = raw_record.numpy()
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(proto_string)
    
    inputs = read_data_proto(proto)
    # inputs_numpy = {k:np.array(v, dtype=object) for k, v in inputs.items()}

    
    inputs_batch = {}
    for k, v in inputs.items():
        if k == 'scenario/id':
            inputs_batch[k] = v
            continue
        if k[:7] == 'traffic':
            inputs_batch[k] = v
            continue
        inputs_batch[k] = np.expand_dims(np.array(v, dtype = np.float32), 0)

    with open('dataset/train_preprocessed_data/{}.pkl'.format(raw_record_index), 'wb') as fp:
        pickle.dump(inputs_batch, fp)
        
    # inputs_batch = add_sdc_fields(inputs_batch)
    # inputs_batch = numpy_to_tf(inputs_batch)
    
    # timestep_grids = create_ground_truth_timestep_grids(inputs=inputs_batch, config=config)
    # true_waypoints = create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=config)
    
    # save_file = {}
    # observed = []
    # occluded = []
    # flow = []
    # flow_origin_occupancy = []

    # for i in range(len(true_waypoints.vehicles.observed_occupancy)):
    #     observed.append(np.expand_dims(np.array(true_waypoints.vehicles.observed_occupancy[i], dtype = np.float), -1))
    #     occluded.append(np.expand_dims(np.array(true_waypoints.vehicles.occluded_occupancy[i], dtype = np.float), -1))
    #     flow.append(np.expand_dims(np.array(true_waypoints.vehicles.flow[i], dtype = np.float), -1))
    #     flow_origin_occupancy.append(np.expand_dims(np.array(true_waypoints.vehicles.flow_origin_occupancy[i], dtype = np.float), -1))
        
    # save_file['scenario/id'] = inputs_batch['scenario/id']
    # save_file["observed"] = np.concatenate(observed, -1)
    # save_file["occluded"] = np.concatenate(occluded, -1)
    # save_file["flow"] = np.concatenate(flow, -1)
    # save_file["flow_origin_occupancy"] = np.concatenate(flow_origin_occupancy, -1)
    
    # # observed = np.concatenate(observed, -1)
    # # occluded = np.concatenate(occluded, -1)
    # # flow = np.concatenate(flow, -1)
    # # flow_origin_occupancy = np.concatenate(flow_origin_occupancy, -1)
    
    # # save_file['gt'] = np.concatenate((observed, occluded, flow, flow_origin_occupancy), -2)

    # with open('dataset/train_gt/tmp_{}.pkl'.format(raw_record_index), 'wb') as fp:
    #     pickle.dump(save_file, fp)
        