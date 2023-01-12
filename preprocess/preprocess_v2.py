from preprocess_tools import *
import pathlib
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib

from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from preprocess_tools import *
from tqdm import tqdm
# import tensorflow_graphics.image.transformer as tfg_transformer

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
import pickle 
from tqdm import tqdm

DATASET_FOLDER = '/nvme/jxs/WaymoOpenMotion/scenario'
# DATASET_FOLDER = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset'

TRAIN_FILES = f'{DATASET_FOLDER}/training/training.tfrecord*'
# TRAIN_FILES = 'training.tfrecord*'
VAL_FILES = f'{DATASET_FOLDER}/validation/validation.tfrecord*'
TEST_FILES = f'{DATASET_FOLDER}/testing/testing.tfrecord*'

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

filenames = tf.io.matching_files(TRAIN_FILES)
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='')
for raw_record_index, raw_record in tqdm(enumerate(raw_dataset)):
    # print(raw_record_index)
    proto_string = raw_record.numpy()
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(proto_string)
    
    inputs = read_data_proto(proto)
    config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
    text_format.Parse(config_text, config)
    
    inputs_batch = {}
    for k, v in inputs.items():
        if k == 'scenario/id':
            inputs_batch[k] = v
            continue
        if k[:7] == 'traffic':
            inputs_batch[k] = v
            continue
        inputs_batch[k] = np.expand_dims(np.array(v, dtype = np.float32), 0)
        
    with open('{}/occupancy_flow/train_preprocessed_data/{}.pkl'.format(DATASET_FOLDER, raw_record_index), 'wb') as fp:
    # with open('{}/train_preprocessed_data/{}.pkl'.format(DATASET_FOLDER, raw_record_index), 'wb') as fp:
        pickle.dump(inputs_batch, fp)
        