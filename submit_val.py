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
from spconv.utils import Point2VoxelCPU3d
from data.data_utils import *
from cumm import tensorview as tv
from models.occupancyflow_field import *
from config import parse_args

generator_agent = Point2VoxelCPU3d(
            vsize_xyz=[1/3.2, 1/3.2, 200],
            coors_range_xyz=[-40, -20, -100, 40, 60, 100],
            num_point_features=5,  # here num_point_features must equal to pc.shape[1]
            max_num_voxels=200000,
            max_num_points_per_voxel=32)
generator_rg = Point2VoxelCPU3d(
            vsize_xyz=[1/3.2, 1/3.2, 200],
            coors_range_xyz=[-40, -20, -100, 40, 60, 100],
            num_point_features=3,  # here num_point_features must equal to pc.shape[1]
            max_num_voxels=200000,
            max_num_points_per_voxel=32)

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
# parse = argparse.ArgumentParser()
# parse.add_argument('--val_preprocessed_path', type = str, default = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/val_preprocessed_data')
# parse.add_argument('--checkpoint_path', type = str, default = './checkpoints/test_2/epoch_10.pth')
# args = parse.parse_args()
args = parse_args()
# val_preprocessed_path = '/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/val_preprocessed_data'
val_files = os.listdir(args.val_preprocessed_file_path)

# with tf.io.gfile.GFile('/nvme/jxs/WaymoOpenMotion/scenario/occupancy_flow/testing_scenario_ids.txt') as f:
#   test_scenario_ids = f.readlines()
#   test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
valid_val_ids_file_path = './valid_val_ids.pkl'




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
  submission.account_name = 'troykalvin78@gmail.com'
  submission.unique_method_name = 'val'
  submission.authors.extend(['Zi Wang'])
  submission.description = 'submit validation'
#   submission.method_link = 'https://www.bing.com/?mkt=zh-CN&mkt=zh-CN'
  return submission

def _add_waypoints_to_scenario_prediction(
    model,
    inputs,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
    """Add predictions for all waypoints to scenario_prediction message."""
    agent_pillars_batch = {}
    rg_pillars_batch = {}
    data = add_sdc(copy.deepcopy(inputs))
    prepared_data  = prepare_for_sample(data, ['past', 'current'])  #translate and rotate according to sdc
    x, y, z, agent_type, agent_valid = sample_agent_points(prepared_data)
    points_x_img, points_y_img, points_is_in_fov = transform_to_image_coordinate(x, y)
    # print(x.shape)     #[1, num_agents, time, sampled_points_num]
    # print(points_is_in_fov.shape)   #[1, num_agents, time, sampled_points_num]
    # print(agent_valid.shape)
    point_is_in_fov_and_valid = np.logical_and(points_is_in_fov, agent_valid.astype(bool))
    points_features_this_batchidx = []
    for idx, object_type in enumerate([1, 2, 3]):  #[1, 2, 3] is all-agent types
        agent_type_matches = np.equal(agent_type, object_type)
        should_keep_point = np.logical_and(point_is_in_fov_and_valid, agent_type_matches)
        # should_keep_point = agent_type_matches
        points_indices = np.argwhere(should_keep_point).astype(np.int32)

        # print(points_indices.shape)
        x_real_coor = x[points_indices[:, 0], points_indices[:, 1], points_indices[:, 2], points_indices[:, 3]]
        y_real_coor = y[points_indices[:, 0], points_indices[:, 1], points_indices[:, 2], points_indices[:, 3]]
        z_real_coor = z[points_indices[:, 0], points_indices[:, 1], points_indices[:, 2], points_indices[:, 3]]
        x_real_coor = np.expand_dims(x_real_coor, axis = -1)
        y_real_coor = np.expand_dims(y_real_coor, axis = -1)
        z_real_coor = np.expand_dims(z_real_coor, axis = -1)
        time_stamp = np.expand_dims(points_indices[:, 2], -1) + 1
        type_concat = np.ones((x_real_coor.shape[0], 1)) * object_type
        # batch_concat = torch.LongTensor([batch_idx+1]).repeat(x_real_coor.shape[0], 1)
        points_features_this_type = np.concatenate((x_real_coor, y_real_coor, z_real_coor, type_concat, time_stamp), -1)

        points_features_this_batchidx.append(points_features_this_type)

    points_features_agent_this_batchidx = np.concatenate(points_features_this_batchidx, 0)
    # print('points_features_agent_this_batchidx', torch.isnan(torch.from_numpy(points_features_agent_this_batchidx)).int().sum())
    # points_features_agent.append(points_features_this_batchidx)
    voxels_agent_this_batchidx, coors_agent_this_batchidx, num_points_per_voxel_agent_this_batchidx = generator_agent.point_to_voxel(tv.from_numpy(points_features_agent_this_batchidx))
    voxels_agent_this_batchidx, coors_agent_this_batchidx, num_points_per_voxel_agent_this_batchidx = torch.from_numpy(voxels_agent_this_batchidx.numpy_view()), torch.from_numpy(coors_agent_this_batchidx.numpy_view()), torch.from_numpy(num_points_per_voxel_agent_this_batchidx.numpy_view())
    # print('voxels_agent_this_batchidx', torch.isnan(voxels_agent_this_batchidx).int().sum())
    # print(voxels_agent_this_batchidx.shape)
    batch_concat = torch.LongTensor([0]).repeat(coors_agent_this_batchidx.shape[0], 1)
    coors_agent_this_batchidx = torch.cat((batch_concat, coors_agent_this_batchidx), -1)

    coors_agent_this_batchidx[:, 2] = 256 - coors_agent_this_batchidx[:, 2]
    x = copy.deepcopy(coors_agent_this_batchidx[:, 3])
    coors_agent_this_batchidx[:, 3] = coors_agent_this_batchidx[:, 1]
    coors_agent_this_batchidx[:, 1] = x
    # print(coors_agent.shape)
    # print(batch_idx, voxels_agent_this_batchidx.shape)
    rg_x, rg_y, rg_z, points_is_in_fov_rg = process_roadgraph(data)
    points_indices = np.argwhere(points_is_in_fov_rg).astype(np.int32)
    rg_x = rg_x[points_indices[:, 0], points_indices[:, 1]]    #[B, N]
    rg_y = rg_y[points_indices[:, 0], points_indices[:, 1]]
    rg_z = rg_z[points_indices[:, 0], points_indices[:, 1]]
    rg_x = rg_x[..., np.newaxis]
    rg_y = rg_y[..., np.newaxis]
    rg_z = rg_z[..., np.newaxis]
    # batch_concat = torch.LongTensor([batch_idx+1]).repeat(rg_x.shape[0], 1)
    points_features_rg_this_batchidx = np.concatenate((rg_x, rg_y, rg_z), -1)
    # print('points_features_rg_this_batchidx', torch.isnan(torch.from_numpy(points_features_rg_this_batchidx)).int().sum())
    voxels_rg_this_batchidx, coors_rg_this_batchidx, num_points_per_voxel_rg_this_batchidx = generator_rg.point_to_voxel(tv.from_numpy(points_features_rg_this_batchidx))
    voxels_rg_this_batchidx, coors_rg_this_batchidx, num_points_per_voxel_rg_this_batchidx = torch.from_numpy(voxels_rg_this_batchidx.numpy_view()), torch.from_numpy(coors_rg_this_batchidx.numpy_view()), torch.from_numpy(num_points_per_voxel_rg_this_batchidx.numpy_view())
    # print('voxels_rg_this_batchidx', torch.isnan(voxels_rg_this_batchidx).int().sum())
    batch_concat = torch.LongTensor([0]).repeat(coors_rg_this_batchidx.shape[0], 1)
    coors_rg_this_batchidx = torch.cat((batch_concat, coors_rg_this_batchidx), -1)

    coors_rg_this_batchidx[:, 2] = 256 - coors_rg_this_batchidx[:, 2]
    x = copy.deepcopy(coors_rg_this_batchidx[:, 3])
    coors_rg_this_batchidx[:, 3] = coors_rg_this_batchidx[:, 1]
    coors_rg_this_batchidx[:, 1] = x
    # print(coors_agent.shape)

    
    agent_pillars_batch['voxels'] = voxels_agent_this_batchidx
    agent_pillars_batch['coors'] = coors_agent_this_batchidx
    agent_pillars_batch['num_points_per_voxel'] = num_points_per_voxel_agent_this_batchidx
    # print('agent_pillars_batch_voxel', torch.isnan(agent_pillars_batch['voxels']).int().sum())
    # print(agent_pillars_batch['voxels'].shape)
    # np.save('/GPFS/rhome/ziwang/projects/occupancy_flow/vis/vis_files/pillar_agent_fast.npy', agent_pillars_batch['coors'])
    
    # print(voxels_rg.shape)
    rg_pillars_batch['voxels'] = voxels_rg_this_batchidx
    rg_pillars_batch['coors'] = coors_rg_this_batchidx
    rg_pillars_batch['num_points_per_voxel'] = num_points_per_voxel_rg_this_batchidx

    with torch.no_grad():
        pred_result = model(agent_pillars_batch, rg_pillars_batch)
    pred_result = pred_result.reshape(1, 256, 256, -1, 8)
    pred_observed_occupancy = pred_result[:, :, :, :1, :]
    pred_occluded_occupancy = pred_result[:, :, :, 1:2, :]
    pred_flow = pred_result[:, :, :, 2:, :]
    waypoints = {}    
    waypoints['observed'] = pred_observed_occupancy.detach().cpu().numpy()
    waypoints['occluded'] = pred_occluded_occupancy.detach().cpu().numpy()
    waypoints['flow'] = pred_flow.detach().cpu().numpy()
    
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
    test_scenario_ids,
    model,
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
          model,
          inputs,
          scenario_prediction=scenario_prediction,
          config=config)

if __name__ == '__main__':
    
    model = OccupancyFlow_Field(args).cuda()
    # model_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    with open(valid_val_ids_file_path, 'rb') as f:
        test_scenario_ids = pickle.load(f)
        
    pathlib.Path('./submission/split').mkdir(exist_ok=True)     
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[:500],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00000-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[500:1000],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00001-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[1000:1500],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00002-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[1500:2000],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00003-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[2000:2500],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00004-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[2500:3000],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00005-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[2500:3000],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00006-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[3000:3500],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00007-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[3500:4000],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00008-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    submission = _make_submission_proto()
    for i, file in tqdm(enumerate(val_files)):
        _generate_predictions_for_one_test_shard(
        test_scenario_ids[4000:4500],
        model,
        submission=submission,
        inputs_file=os.path.join(args.val_preprocessed_file_path, file),
        file_name = file,
        shard_message=f'{i + 1} of {len(val_files)}')
    submission_file_path = './submission/split/submit_validation'
    f = open(submission_file_path + '.binproto-00009-of-00009', 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    # import tarfile
    # with tarfile.open(submission_file_path+".tar.gz", "w:gz") as tar:
    #     tar.add(submission_file_path + '.binproto')

    #tar czvf ./submission/split/submit_validation.tar.gz -C ./submission/split .