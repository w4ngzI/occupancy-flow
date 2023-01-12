import submission.occupancy_flow_submission_pb2 as occupancy_flow_submission_pb2
import submission.occupancy_flow_metrics_pb2 as occupancy_flow_metrics_pb2
import pathlib
import zlib
import os
from tqdm import tqdm
import numpy as np
from google.protobuf import text_format
import pickle
from data.gt_utils import *
import argparse
import os

config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
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
text_format.Parse(config_text, config)
parse = argparse.ArgumentParser()
parse.add_argument('--val_preprocessed_path', type = str, default = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/val_preprocessed_data')
args = parse.parse_args()
# val_preprocessed_path = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/val_preprocessed_data'
val_files = os.listdir(args.val_preprocessed_path)

# with tf.io.gfile.GFile('/nvme/jxs/WaymoOpenMotion/scenario/occupancy_flow/testing_scenario_ids.txt') as f:
#   test_scenario_ids = f.readlines()
#   test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
valid_val_ids_file_path = './valid_val_ids.pkl'

with open(valid_val_ids_file_path, 'rb') as f:
    test_scenario_ids = pickle.load(f)
    
def add_sdc(data):
    sdc_indices = np.argwhere(data['state/is_sdc'] == True)
    # print(sdc_indices.shape)
    sdc_indices = np.squeeze(sdc_indices, 0)
    # print(sdc_indices)
    data['sdc/current/x'] = np.expand_dims(data['state/current/x'][sdc_indices[0], sdc_indices[1]], 0)
    data['sdc/current/y'] = np.expand_dims(data['state/current/y'][sdc_indices[0], sdc_indices[1]], 0)
    data['sdc/current/z'] = np.expand_dims(data['state/current/z'][sdc_indices[0], sdc_indices[1]], 0)
    data['sdc/current/bbox_yaw'] = np.expand_dims(data['state/current/bbox_yaw'][sdc_indices[0], sdc_indices[1]], 0)
    return data
  
def _make_submission_proto(
) -> occupancy_flow_submission_pb2.ChallengeSubmission:
  """Makes a submission proto to store predictions for one shard."""
  submission = occupancy_flow_submission_pb2.ChallengeSubmission()
  submission.account_name = 'w4ngz1@sjtu.edu.cn'
  submission.unique_method_name = 'val gt'
  submission.authors.extend(['Zi Wang'])
  submission.description = 'submit validation ground truth'
#   submission.method_link = 'https://www.bing.com/?mkt=zh-CN&mkt=zh-CN'
  return submission

def _add_waypoints_to_scenario_prediction(
    inputs,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Add predictions for all waypoints to scenario_prediction message."""
  inputs_gt = add_sdc(copy.deepcopy(inputs))
  timestep_grids = create_ground_truth_timestep_grids(inputs=inputs_gt, config=config)
  true_waypoints = create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=config)
  
  observed_tmp = []
  occluded_tmp = []
  flow_tmp = []
  waypoints = {}
  
  for i in range(len(true_waypoints.vehicles.observed_occupancy)):
    observed_tmp.append(np.expand_dims(true_waypoints.vehicles.observed_occupancy[i], -1))
    occluded_tmp.append(np.expand_dims(true_waypoints.vehicles.occluded_occupancy[i], -1))
    flow_tmp.append(np.expand_dims(true_waypoints.vehicles.flow[i], -1))
    
  waypoints['observed'] = np.concatenate(observed_tmp, -1)
  waypoints['occluded'] = np.concatenate(occluded_tmp, -1)
  waypoints['flow'] = np.concatenate(flow_tmp, -1)
  
  for k in range(config.num_waypoints):
    waypoint_message = scenario_prediction.waypoints.add()
    # Observed occupancy.
    # obs_occupancy = pred_waypoints.vehicles.observed_occupancy[k].numpy()
    obs_occupancy = waypoints['observed'][..., k]
    # print('obs_occupancy', obs_occupancy.shape)
    obs_occupancy_quantized = np.round(obs_occupancy * 255).astype(np.uint8)
    obs_occupancy_bytes = zlib.compress(obs_occupancy_quantized.tobytes())
    waypoint_message.observed_vehicles_occupancy = obs_occupancy_bytes
    # Occluded occupancy.
    # occ_occupancy = pred_waypoints.vehicles.occluded_occupancy[k].numpy()
    occ_occupancy = waypoints['occluded'][..., k]
    # print('occ_occupancy', occ_occupancy.shape)
    occ_occupancy_quantized = np.round(occ_occupancy * 255).astype(np.uint8)
    occ_occupancy_bytes = zlib.compress(occ_occupancy_quantized.tobytes())
    waypoint_message.occluded_vehicles_occupancy = occ_occupancy_bytes
    # Flow.
    # flow = pred_waypoints.vehicles.flow[k].numpy()
    flow = waypoints['flow'][..., k]
    # print('flow.shape', flow.shape)
    flow_quantized = np.clip(np.round(flow), -128, 127).astype(np.int8)
    flow_bytes = zlib.compress(flow_quantized.tobytes())
    waypoint_message.all_vehicles_flow = flow_bytes

def _generate_predictions_for_one_test_shard(
    submission: occupancy_flow_submission_pb2.ChallengeSubmission,
    inputs_file,
    file_name,
    shard_message,
) -> None:
    
    """Iterate over all test examples in one shard and generate predictions."""
    if file_name in test_scenario_ids:
      inputs = np.load(inputs_file, allow_pickle = True)
    
      # print(f'Processing test shard {shard_message}')
      # Run inference.
    #   pred_waypoint_logits = _run_model_on_inputs(inputs=inputs, training=False)
    #   pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

      # Make new scenario prediction message.
      scenario_prediction = submission.scenario_predictions.add()
      scenario_prediction.scenario_id = inputs['scenario/id']
      # print(scenario_prediction.scenario_id)

      # Add all waypoints.
      _add_waypoints_to_scenario_prediction(
          inputs,
          scenario_prediction=scenario_prediction,
          config=config)

if __name__ == '__main__':
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        
        _generate_predictions_for_one_test_shard(
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
        
    # _save_submission_to_file(submission=submission)
    submission_file_path = './submission/submit_validation_gt'
    f = open(submission_file_path + '.binproto', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    import tarfile
    with tarfile.open(submission_file_path+".tar.gz", "w:gz") as tar:
        tar.add(submission_file_path + '.binproto')

    