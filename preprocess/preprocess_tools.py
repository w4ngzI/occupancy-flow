import dataclasses
import functools
from typing import List, Mapping, Optional, Sequence, Tuple, Dict, Union
import numpy as np
import tensorflow as tf
import math
import uuid
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
from copy import deepcopy


_ObjectType = scenario_pb2.Track.ObjectType
ALL_AGENT_TYPES = [
    _ObjectType.TYPE_VEHICLE,
    _ObjectType.TYPE_PEDESTRIAN,
    _ObjectType.TYPE_CYCLIST,
]

@dataclasses.dataclass
class AgentGrids:
  """Contains any topdown render for vehicles and pedestrians."""
  vehicles: Optional[tf.Tensor] = None
  pedestrians: Optional[tf.Tensor] = None
  cyclists: Optional[tf.Tensor] = None

  def view(self, agent_type: str) -> tf.Tensor:
    """Retrieve topdown tensor for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')
    
@dataclasses.dataclass
class _TimestepGridsOneType:
  """Occupancy and flow tensors over past/current/future for one agent type."""
  # [batch_size, height, width, 1]
  current_occupancy: Optional[tf.Tensor] = None
  # [batch_size, height, width, num_past_steps]
  past_occupancy: Optional[tf.Tensor] = None
  # [batch_size, height, width, num_future_steps]
  future_observed_occupancy: Optional[tf.Tensor] = None
  # [batch_size, height, width, num_future_steps]
  future_occluded_occupancy: Optional[tf.Tensor] = None
  # Backward flow (dx, dy) for all observed and occluded agents.  Flow is
  # constructed between timesteps `waypoint_size` apart over all timesteps in
  # [past, current, future].  The flow for each timestep `t` contains (dx, dy)
  # occupancy displacements from timestep `t` to timestep `t - waypoints_size`,
  # which is EARLIER than t (hence backward flow).
  # waypoint_size = num_future_steps // num_waypoints
  # num_flow_steps = (num_past_steps + 1 + num_future_steps) - waypoint_size
  # [batch_size, height, width, num_flow_steps, 2]
  all_flow: Optional[tf.Tensor] = None
  # Observed and occluded occupancy over all timesteps.  This is used to
  # generate flow_origin tensors in WaypointGrids.
  # [batch_size, height, width, num_past_steps + 1 + num_future_steps]
  all_occupancy: Optional[tf.Tensor] = None


# Holds ground-truth occupancy and flow tensors for each timestep in
# past/current/future for all agent classes.
@dataclasses.dataclass
class TimestepGrids:
  """Occupancy and flow for vehicles, pedestrians, cyclists."""
  vehicles: _TimestepGridsOneType = dataclasses.field(
      default_factory=_TimestepGridsOneType)
  pedestrians: _TimestepGridsOneType = dataclasses.field(
      default_factory=_TimestepGridsOneType)
  cyclists: _TimestepGridsOneType = dataclasses.field(
      default_factory=_TimestepGridsOneType)

  def view(self, agent_type: str) -> _TimestepGridsOneType:
    """Retrieve occupancy and flow tensors for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')


# Holds num_waypoints occupancy and flow tensors for one agent class.
@dataclasses.dataclass
class _WaypointGridsOneType:
  """Sequence of num_waypoints occupancy and flow tensors for one agent type."""
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  observed_occupancy: List[tf.Tensor] = dataclasses.field(default_factory=list)
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  occluded_occupancy: List[tf.Tensor] = dataclasses.field(default_factory=list)
  # num_waypoints tensors shaped [batch_size, height, width, 2]
  flow: List[tf.Tensor] = dataclasses.field(default_factory=list)
  # The origin occupancy for each flow waypoint.  Notice that a flow field
  # transforms some origin occupancy into some destination occupancy.
  # Flow-origin occupancies are the base occupancies for each flow field.
  # num_waypoints tensors shaped [batch_size, height, width, 1]
  flow_origin_occupancy: List[tf.Tensor] = dataclasses.field(
      default_factory=list)


# Holds num_waypoints occupancy and flow tensors for all agent clases.  This is
# used to store both ground-truth and predicted topdowns.
@dataclasses.dataclass
class WaypointGrids:
  """Occupancy and flow sequences for vehicles, pedestrians, cyclists."""
  vehicles: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)
  pedestrians: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)
  cyclists: _WaypointGridsOneType = dataclasses.field(
      default_factory=_WaypointGridsOneType)

  def view(self, agent_type: str) -> _WaypointGridsOneType:
    """Retrieve occupancy and flow sequences for given agent type."""
    if agent_type == _ObjectType.TYPE_VEHICLE:
      return self.vehicles
    elif agent_type == _ObjectType.TYPE_PEDESTRIAN:
      return self.pedestrians
    elif agent_type == _ObjectType.TYPE_CYCLIST:
      return self.cyclists
    else:
      raise ValueError(f'Unknown agent type:{agent_type}.')

  def get_observed_occupancy_at_waypoint(
      self, k: int) -> AgentGrids:
    """Returns occupancies of currently-observed agents at waypoint k."""
    agent_grids = AgentGrids()
    if self.vehicles.observed_occupancy:
      agent_grids.vehicles = self.vehicles.observed_occupancy[k]
    if self.pedestrians.observed_occupancy:
      agent_grids.pedestrians = self.pedestrians.observed_occupancy[k]
    if self.cyclists.observed_occupancy:
      agent_grids.cyclists = self.cyclists.observed_occupancy[k]
    return agent_grids

  def get_occluded_occupancy_at_waypoint(
      self, k: int) -> AgentGrids:
    """Returns occupancies of currently-occluded agents at waypoint k."""
    agent_grids = AgentGrids()
    if self.vehicles.occluded_occupancy:
      agent_grids.vehicles = self.vehicles.occluded_occupancy[k]
    if self.pedestrians.occluded_occupancy:
      agent_grids.pedestrians = self.pedestrians.occluded_occupancy[k]
    if self.cyclists.occluded_occupancy:
      agent_grids.cyclists = self.cyclists.occluded_occupancy[k]
    return agent_grids

  def get_flow_at_waypoint(self, k: int) -> AgentGrids:
    """Returns flow fields of all agents at waypoint k."""
    agent_grids = AgentGrids()
    if self.vehicles.flow:
      agent_grids.vehicles = self.vehicles.flow[k]
    if self.pedestrians.flow:
      agent_grids.pedestrians = self.pedestrians.flow[k]
    if self.cyclists.flow:
      agent_grids.cyclists = self.cyclists.flow[k]
    return agent_grids


# Holds topdown renders of scene objects suitable for visualization.
@dataclasses.dataclass
class VisGrids:
  # Roadgraph elements.
  # [batch_size, height, width, 1]
  roadgraph: Optional[tf.Tensor] = None
  # Trail of scene agents over past and current time frames.
  # [batch_size, height, width, 1]
  agent_trails: Optional[tf.Tensor] = None
  
@dataclasses.dataclass
class _SampledPoints:
  """Set of points sampled from agent boxes.

  All fields have shape -
  [batch_size, num_agents, num_steps, num_points] where num_points is
  (points_per_side_length * points_per_side_width).
  """
  # [batch, num_agents, num_steps, points_per_agent].
  x: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  y: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  z: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  agent_type: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  valid: tf.Tensor

  
  
def create_ground_truth_timestep_grids(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> TimestepGrids:
  """Renders topdown views of agents over past/current/future time frames.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    TimestepGrids object holding topdown renders of agents.
  """
  timestep_grids = TimestepGrids()

  # Occupancy grids.
  render_func = functools.partial(
      render_occupancy_from_inputs,
      inputs=inputs,
      config=config)

  current_occupancy = render_func(
      times=['current'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.current_occupancy = current_occupancy.vehicles
  timestep_grids.pedestrians.current_occupancy = current_occupancy.pedestrians
  timestep_grids.cyclists.current_occupancy = current_occupancy.cyclists

  past_occupancy = render_func(
      times=['past'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.past_occupancy = past_occupancy.vehicles
  timestep_grids.pedestrians.past_occupancy = past_occupancy.pedestrians
  timestep_grids.cyclists.past_occupancy = past_occupancy.cyclists

  future_obs = render_func(
      times=['future'],
      include_observed=True,
      include_occluded=False,
  )
  timestep_grids.vehicles.future_observed_occupancy = future_obs.vehicles
  timestep_grids.pedestrians.future_observed_occupancy = future_obs.pedestrians
  timestep_grids.cyclists.future_observed_occupancy = future_obs.cyclists

  future_occ = render_func(
      times=['future'],
      include_observed=False,
      include_occluded=True,
  )
  timestep_grids.vehicles.future_occluded_occupancy = future_occ.vehicles
  timestep_grids.pedestrians.future_occluded_occupancy = future_occ.pedestrians
  timestep_grids.cyclists.future_occluded_occupancy = future_occ.cyclists

  # All occupancy for flow_origin_occupancy.
  all_occupancy = render_func(
      times=['past', 'current', 'future'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.all_occupancy = all_occupancy.vehicles
  timestep_grids.pedestrians.all_occupancy = all_occupancy.pedestrians
  timestep_grids.cyclists.all_occupancy = all_occupancy.cyclists

  # Flow.
  # NOTE: Since the future flow depends on the current and past timesteps, we
  # need to compute it from [past + current + future] sparse points.
  all_flow = render_flow_from_inputs(
      inputs=inputs,
      times=['past', 'current', 'future'],
      config=config,
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.all_flow = all_flow.vehicles
  timestep_grids.pedestrians.all_flow = all_flow.pedestrians
  timestep_grids.cyclists.all_flow = all_flow.cyclists

  return timestep_grids


def create_ground_truth_waypoint_grids(
    timestep_grids: TimestepGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> WaypointGrids:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    WaypointGrids object.
  """
  if config.num_future_steps % config.num_waypoints != 0:
    raise ValueError(f'num_future_steps({config.num_future_steps}) must be '
                     f'a multiple of num_waypoints({config.num_waypoints}).')

  true_waypoints = WaypointGrids(
      vehicles=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
      pedestrians=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
      cyclists=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
  )

  # Observed occupancy.
  _add_ground_truth_observed_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Occluded occupancy.
  _add_ground_truth_occluded_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Flow origin occupancy.
  _add_ground_truth_flow_origin_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Flow.
  _add_ground_truth_flow_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)

  return true_waypoints


def _add_ground_truth_observed_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  for object_type in ALL_AGENT_TYPES:
    # [batch_size, height, width, num_future_steps]
    future_obs = timestep_grids.view(object_type).future_observed_occupancy
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = future_obs[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_occupancy = future_obs[..., waypoint_end - 1:waypoint_end]
      waypoint_grids.view(object_type).observed_occupancy.append(
          waypoint_occupancy)


def _add_ground_truth_occluded_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  for object_type in ALL_AGENT_TYPES:
    # [batch_size, height, width, num_future_steps]
    future_occ = timestep_grids.view(object_type).future_occluded_occupancy
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = future_occ[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_occupancy = future_occ[..., waypoint_end - 1:waypoint_end]
      waypoint_grids.view(object_type).occluded_occupancy.append(
          waypoint_occupancy)


def _add_ground_truth_flow_origin_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates topdowns as origin occupancies for flow fields.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  num_history_steps = config.num_past_steps + 1  # Includes past + current.
  num_future_steps = config.num_future_steps
  if waypoint_size > num_history_steps:
    raise ValueError('If waypoint_size > num_history_steps, we cannot find the '
                     'flow origin occupancy for the first waypoint.')

  for object_type in ALL_AGENT_TYPES:
    # [batch_size, height, width, num_past_steps + 1 + num_future_steps]
    all_occupancy = timestep_grids.view(object_type).all_occupancy
    # Keep only the section containing flow_origin_occupancy timesteps.
    # First remove `waypoint_size` from the end.  Then keep the tail containing
    # num_future_steps timesteps.
    flow_origin_occupancy = all_occupancy[:, :, :, :-waypoint_size]
    # [batch_size, height, width, num_future_steps]
    flow_origin_occupancy = flow_origin_occupancy[:, :, :, -num_future_steps:]
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = flow_origin_occupancy[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_flow_origin = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_flow_origin = flow_origin_occupancy[..., waypoint_end -
                                                     1:waypoint_end]
      waypoint_grids.view(object_type).flow_origin_occupancy.append(
          waypoint_flow_origin)


def _add_ground_truth_flow_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future flow fields as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  num_future_steps = config.num_future_steps
  waypoint_size = config.num_future_steps // config.num_waypoints

  for object_type in ALL_AGENT_TYPES:
    # num_flow_steps = (num_past_steps + num_futures_steps) - waypoint_size
    # [batch_size, height, width, num_flow_steps, 2]
    flow = timestep_grids.view(object_type).all_flow
    # Keep only the flow tail, containing num_future_steps timesteps.
    # [batch_size, height, width, num_future_steps, 2]
    flow = flow[:, :, :, -num_future_steps:, :]
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size, 2]
        segment = flow[:, :, :, waypoint_start:waypoint_end, :]
        # Compute mean flow over the timesteps in this segment by counting
        # the number of pixels with non-zero flow and dividing the flow sum
        # by that number.
        # [batch_size, height, width, waypoint_size, 2]
        occupied_pixels = tf.cast(tf.not_equal(segment, 0.0), tf.float32)
        # [batch_size, height, width, 2]
        num_flow_values = tf.reduce_sum(occupied_pixels, axis=3)
        # [batch_size, height, width, 2]
        segment_sum = tf.reduce_sum(segment, axis=3)
        # [batch_size, height, width, 2]
        mean_flow = tf.math.divide_no_nan(segment_sum, num_flow_values)
        waypoint_flow = mean_flow
      else:
        waypoint_flow = flow[:, :, :, waypoint_end - 1, :]
      waypoint_grids.view(object_type).flow.append(waypoint_flow)


def create_ground_truth_vis_grids(
    inputs: Mapping[str, tf.Tensor],
    timestep_grids: TimestepGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> VisGrids:
  """Creates topdown renders of roadgraph and agent trails for visualization.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    timestep_grids: Holds topdown renders of agents over time.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    VisGrids object holding roadgraph and agent trails over past, current time.
  """
  roadgraph = render_roadgraph_from_inputs(
      inputs, config)
  agent_trails = _create_agent_trails(timestep_grids)

  return VisGrids(
      roadgraph=roadgraph,
      agent_trails=agent_trails,
  )


def _create_agent_trails(
    timestep_grids: TimestepGrids,
    gamma: float = 0.80,
) -> tf.Tensor:
  """Renders trails for all agents over the past and current time frames.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    gamma: Decay for older boxes.

  Returns:
    Agent trails as [batch_size, height, width, 1] float32 tensor.
  """
  agent_trails = 0.0
  num_past = tf.convert_to_tensor(
      timestep_grids.vehicles.past_occupancy).shape.as_list()[-1]
  for i in range(num_past):
    new_agents = (
        timestep_grids.vehicles.past_occupancy[..., i:i + 1] +
        timestep_grids.pedestrians.past_occupancy[..., i:i + 1] +
        timestep_grids.cyclists.past_occupancy[..., i:i + 1])
    agent_trails = tf.clip_by_value(agent_trails * gamma + new_agents, 0, 1)
  new_agents = (
      timestep_grids.vehicles.current_occupancy +
      timestep_grids.pedestrians.current_occupancy +
      timestep_grids.cyclists.current_occupancy)
  agent_trails = tf.clip_by_value(agent_trails * gamma * gamma + new_agents, 0,
                                  1)
  return agent_trails

def render_occupancy_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> AgentGrids:
  """Creates topdown renders of agents grouped by agent class.

  Renders agent boxes by densely sampling points from their boxes.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    An AgentGrids object containing:
      vehicles: [batch_size, height, width, steps] float32 in [0, 1].
      pedestrians: [batch_size, height, width, steps] float32 in [0, 1].
      cyclists: [batch_size, height, width, steps] float32 in [0, 1].
      where steps is the number of timesteps covered in `times`.
  """
  sampled_points = _sample_and_filter_agent_points(
      inputs=inputs,
      times=times,
      config=config,
      include_observed=include_observed,
      include_occluded=include_occluded,
  )

  agent_x = sampled_points.x
  agent_y = sampled_points.y
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  topdown_shape = [
      batch_size, config.grid_height_cells, config.grid_width_cells, num_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov,
                  [batch_size, num_agents, num_steps, points_per_agent])])

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)
  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  occupancies = {}
  for object_type in ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.

    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)

    assert_shapes([
        (should_render_point,
         [batch_size, num_agents, num_steps, points_per_agent]),
    ])

    # Scatter points across topdown maps for each timestep.  The tensor
    # `point_indices` holds the indices where `should_render_point` is True.
    # It is a 2-D tensor with shape [n, 4], where n is the number of valid
    # agent points inside FOV.  Each row in this tensor contains indices over
    # the following 4 dimensions: (batch, agent, timestep, point).

    # [num_points_to_render, 4]
    point_indices = tf.cast(tf.where(should_render_point), tf.int32)
    # [num_points_to_render, 1]
    x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
    y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

    num_points_to_render = point_indices.shape.as_list()[0]
    assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                   (y_img_coord, [num_points_to_render, 1])])

    # [num_points_to_render, 4]
    xy_img_coord = tf.concat(
        [
            point_indices[:, :1],
            tf.cast(y_img_coord, tf.int32),
            tf.cast(x_img_coord, tf.int32),
            point_indices[:, 2:3],
        ],
        axis=1,
    )
    # [num_points_to_render]
    gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

    # [batch_size, grid_height_cells, grid_width_cells, num_steps]
    topdown = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
    assert_shapes([(topdown, topdown_shape)])

    # scatter_nd() accumulates values if there are repeated indices.  Since
    # we sample densely, this happens all the time.  Clip the final values.
    topdown = tf.clip_by_value(topdown, 0.0, 1.0)
    occupancies[object_type] = topdown
  # print(_ObjectType.TYPE_VEHICLE)
  return AgentGrids(
      vehicles=occupancies[_ObjectType.TYPE_VEHICLE],
      pedestrians=occupancies[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=occupancies[_ObjectType.TYPE_CYCLIST],
  )


def render_flow_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> AgentGrids:
  """Compute top-down flow between timesteps `waypoint_size` apart.

  Returns (dx, dy) for each timestep.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    An AgentGrids object containing:
      vehicles: [batch_size, height, width, num_flow_steps, 2] float32
      pedestrians: [batch_size, height, width, num_flow_steps, 2] float32
      cyclists: [batch_size, height, width, num_flow_steps, 2] float32
      where num_flow_steps = num_steps - waypoint_size, and num_steps is the
      number of timesteps covered in `times`.
  """
  sampled_points = _sample_and_filter_agent_points(
      inputs=inputs,
      times=times,
      config=config,
      include_observed=include_observed,
      include_occluded=include_occluded,
  )

  agent_x = sampled_points.x
  agent_y = sampled_points.y
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  # The timestep distance between flow steps.
  waypoint_size = config.num_future_steps // config.num_waypoints
  num_flow_steps = num_steps - waypoint_size
  topdown_shape = [
      batch_size, config.grid_height_cells, config.grid_width_cells,
      num_flow_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov,
                  [batch_size, num_agents, num_steps, points_per_agent])])

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)

  # Backward Flow.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  dx = agent_x[:, :, :-waypoint_size, :] - agent_x[:, :, waypoint_size:, :]
  dy = agent_y[:, :, :-waypoint_size, :] - agent_y[:, :, waypoint_size:, :]
  assert_shapes([
      (dx, [batch_size, num_agents, num_flow_steps, points_per_agent]),
      (dy, [batch_size, num_agents, num_flow_steps, points_per_agent]),
  ])

  # Adjust other fields as well to reduce from num_steps to num_flow_steps.
  # agent_x, agent_y: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_x = agent_x[:, :, waypoint_size:, :]
  agent_y = agent_y[:, :, waypoint_size:, :]
  # agent_type: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_type = agent_type[:, :, waypoint_size:, :]
  # point_is_in_fov: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  point_is_in_fov = point_is_in_fov[:, :, waypoint_size:, :]
  # agent_valid: And the two timesteps.  They both need to be valid.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_valid = tf.logical_and(agent_valid[:, :, waypoint_size:, :],
                               agent_valid[:, :, :-waypoint_size, :])

  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  flows = {}
  for object_type in ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)
    assert_shapes([
        (should_render_point,
         [batch_size, num_agents, num_flow_steps, points_per_agent]),
    ])

    # [batch_size, height, width, num_flow_steps, 2]
    flow = _render_flow_points_for_one_agent_type(
        agent_x=agent_x,
        agent_y=agent_y,
        dx=dx,
        dy=dy,
        should_render_point=should_render_point,
        topdown_shape=topdown_shape,
    )
    flows[object_type] = flow

  return AgentGrids(
      vehicles=flows[_ObjectType.TYPE_VEHICLE],
      pedestrians=flows[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=flows[_ObjectType.TYPE_CYCLIST],
  )


def _render_flow_points_for_one_agent_type(
    agent_x: tf.Tensor,
    agent_y: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    should_render_point: tf.Tensor,
    topdown_shape: List[int],
) -> tf.Tensor:
  """Renders topdown (dx, dy) flow for given agent points.

  Args:
    agent_x: [batch_size, num_agents, num_steps, points_per_agent].
    agent_y: [batch_size, num_agents, num_steps, points_per_agent].
    dx: [batch_size, num_agents, num_steps, points_per_agent].
    dy: [batch_size, num_agents, num_steps, points_per_agent].
    should_render_point: [batch_size, num_agents, num_steps, points_per_agent].
    topdown_shape: Shape of the output flow field.

  Returns:
    Rendered flow as [batch_size, height, width, num_flow_steps, 2] float32
      tensor.
  """
  assert_shapes = tf.debugging.assert_shapes

  # Scatter points across topdown maps for each timestep.  The tensor
  # `point_indices` holds the indices where `should_render_point` is True.
  # It is a 2-D tensor with shape [n, 4], where n is the number of valid
  # agent points inside FOV.  Each row in this tensor contains indices over
  # the following 4 dimensions: (batch, agent, timestep, point).

  # [num_points_to_render, 4]
  point_indices = tf.cast(tf.where(should_render_point), tf.int32)
  # [num_points_to_render, 1]
  x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
  y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

  num_points_to_render = point_indices.shape.as_list()[0]
  assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                 (y_img_coord, [num_points_to_render, 1])])

  # [num_points_to_render, 4]
  xy_img_coord = tf.concat(
      [
          point_indices[:, :1],
          tf.cast(y_img_coord, tf.int32),
          tf.cast(x_img_coord, tf.int32),
          point_indices[:, 2:3],
      ],
      axis=1,
  )
  # [num_points_to_render]
  gt_values_dx = tf.gather_nd(dx, point_indices)
  gt_values_dy = tf.gather_nd(dy, point_indices)

  # tf.scatter_nd() accumulates values when there are repeated indices.
  # Keep track of number of indices writing to the same pixel so we can
  # account for accumulated values.
  # [num_points_to_render]
  gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps]
  flow_x = tf.scatter_nd(xy_img_coord, gt_values_dx, topdown_shape)
  flow_y = tf.scatter_nd(xy_img_coord, gt_values_dy, topdown_shape)
  num_values_per_pixel = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
  assert_shapes([
      (flow_x, topdown_shape),
      (flow_y, topdown_shape),
      (num_values_per_pixel, topdown_shape),
  ])

  # Undo the accumulation effect of tf.scatter_nd() for repeated indices.
  flow_x = tf.math.divide_no_nan(flow_x, num_values_per_pixel)
  flow_y = tf.math.divide_no_nan(flow_y, num_values_per_pixel)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps, 2]
  flow = tf.stack([flow_x, flow_y], axis=-1)
  assert_shapes([(flow, topdown_shape + [2])])
  return flow

def render_roadgraph_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> tf.Tensor:
  """Creates a topdown render of roadgraph points.

  This function is mostly useful for visualization.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    Rendered roadgraph as [batch_size, height, width, 1] float32 tensor
      containing zeros and ones.
  """
  grid_height_cells = config.grid_height_cells
  grid_width_cells = config.grid_width_cells

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_rg_points, _ = (
      inputs['roadgraph_samples/xyz'].shape.as_list())
  topdown_shape = [batch_size, grid_height_cells, grid_width_cells, 1]

  # Translate the roadgraph points so that the autonomous vehicle is at the
  # origin.
  sdc_xyz = tf.concat(
      [
          inputs['sdc/current/x'],
          inputs['sdc/current/y'],
          inputs['sdc/current/z'],
      ],
      axis=1,
  )
  # [batch_size, 1, 3]
  sdc_xyz = sdc_xyz[:, tf.newaxis, :]
  # [batch_size, num_rg_points, 3]
  rg_points = inputs['roadgraph_samples/xyz'] - sdc_xyz

  # [batch_size, num_rg_points, 1]
  # rg_valid = inputs['roadgraph_samples/valid']
  # assert_shapes([(rg_points, [batch_size, num_rg_points, 3]),
  #                (rg_valid, [batch_size, num_rg_points, 1])])
  assert_shapes([(rg_points, [batch_size, num_rg_points, 3])])
  # [batch_size, num_rg_points]
  rg_x, rg_y, _ = tf.unstack(rg_points, axis=-1)
  assert_shapes([(rg_x, [batch_size, num_rg_points]),
                 (rg_y, [batch_size, num_rg_points])])

  if config.normalize_sdc_yaw:
    angle = math.pi / 2 - inputs['sdc/current/bbox_yaw']
    rg_x, rg_y = rotate_points_around_origin(rg_x, rg_y, angle)

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_rg_points]
  rg_x, rg_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=rg_x,
      points_y=rg_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov, [batch_size, num_rg_points])])

  # Filter out invalid points.
  # point_is_valid = tf.cast(rg_valid[..., 0], tf.bool)
  # [batch, num_rg_points]
  # should_render_point = tf.logical_and(point_is_in_fov, point_is_valid)
  should_render_point = point_is_in_fov

  # Scatter points across a topdown map.  The tensor `point_indices` holds the
  # indices where `should_render_point` is True.  It is a 2-D tensor with shape
  # [n, 2], where n is the number of valid roadgraph points inside FOV.  Each
  # row in this tensor contains indices over the following 2 dimensions:
  # (batch, point).

  # [num_points_to_render, 2] holding (batch index, point index).
  point_indices = tf.cast(tf.where(should_render_point), tf.int32)
  # [num_points_to_render, 1]
  x_img_coord = tf.gather_nd(rg_x, point_indices)[..., tf.newaxis]
  y_img_coord = tf.gather_nd(rg_y, point_indices)[..., tf.newaxis]

  num_points_to_render = point_indices.shape.as_list()[0]
  assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                 (y_img_coord, [num_points_to_render, 1])])

  # [num_points_to_render, 3]
  xy_img_coord = tf.concat(
      [
          point_indices[:, :1],
          tf.cast(y_img_coord, tf.int32),
          tf.cast(x_img_coord, tf.int32),
      ],
      axis=1,
  )
  # Set pixels with roadgraph points to 1.0, leave others at 0.0.
  # [num_points_to_render, 1]
  gt_values = tf.ones_like(x_img_coord, dtype=tf.float32)

  # [batch_size, grid_height_cells, grid_width_cells, 1]
  rg_viz = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
  assert_shapes([(rg_viz, topdown_shape)])

  # scatter_nd() accumulates values if there are repeated indices.  Clip the
  # final values to handle cases where two roadgraph points coincide.
  rg_viz = tf.clip_by_value(rg_viz, 0.0, 1.0)
  return rg_viz


def _transform_to_image_coordinates(
    points_x: tf.Tensor,
    points_y: tf.Tensor,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns transformed points and a mask indicating whether point is in image.

  Args:
    points_x: Tensor of any shape containing x values in world coordinates
      centered on the autonomous vehicle (see translate_sdc_to_origin).
    points_y: Tensor with same shape as points_x containing y values in world
      coordinates centered on the autonomous vehicle.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    Tuple containing the following tensors:
      - Transformed points_x.
      - Transformed points_y.
      - tf.bool tensor with same shape as points_x indicating which points are
        inside the FOV of the image after transformation.
  """
  pixels_per_meter = config.pixels_per_meter
  points_x = tf.round(points_x * pixels_per_meter) + config.sdc_x_in_grid
  points_y = tf.round(-points_y * pixels_per_meter) + config.sdc_y_in_grid

  # Filter out points that are located outside the FOV of topdown map.
  point_is_in_fov = tf.logical_and(
      tf.logical_and(
          tf.greater_equal(points_x, 0), tf.greater_equal(points_y, 0)),
      tf.logical_and(
          tf.less(points_x, config.grid_width_cells),
          tf.less(points_y, config.grid_height_cells)))

  return points_x, points_y, point_is_in_fov


def _sample_and_filter_agent_points(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> _SampledPoints:
  """Samples points and filters them according to current visibility of agents.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    _SampledPoints: containing x, y, z coordinates, type, and valid bits.
  """
  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, _ = (inputs['state/current/x'].shape.as_list())
  num_steps = _get_num_steps_from_times(times, config)
  points_per_agent = (
      config.agent_points_per_side_length * config.agent_points_per_side_width)

  # Sample points from agent boxes over specified time frames.
  # All fields have shape [batch_size, num_agents, num_steps, points_per_agent].
  sampled_points = _sample_agent_points(
      inputs,
      times=times,
      points_per_side_length=config.agent_points_per_side_length,
      points_per_side_width=config.agent_points_per_side_width,
      translate_sdc_to_origin=True,
      normalize_sdc_yaw=config.normalize_sdc_yaw,
  )

  field_shape = [batch_size, num_agents, num_steps, points_per_agent]
  assert_shapes([
      (sampled_points.x, field_shape),
      (sampled_points.y, field_shape),
      (sampled_points.z, field_shape),
      (sampled_points.valid, field_shape),
      (sampled_points.agent_type, field_shape),
  ])

  agent_valid = tf.cast(sampled_points.valid, tf.bool)
  # 1. If all agents are requested, no additional filtering is necessary.
  # 2. Filter observed/occluded agents for future only.
  include_all = include_observed and include_occluded
  if not include_all and 'future' in times:
    history_times = ['past', 'current']
    # [batch_size, num_agents, num_history_steps, 1]
    agent_is_observed = _stack_field(inputs, history_times, 'valid')
    # [batch_size, num_agents, 1, 1]
    agent_is_observed = tf.reduce_max(agent_is_observed, axis=2, keepdims=True)  #np.max
    agent_is_observed = tf.cast(agent_is_observed, tf.bool)

    if include_observed:
      agent_filter = agent_is_observed
    elif include_occluded:
      agent_filter = tf.logical_not(agent_is_observed)
    else:  # Both observed and occluded are off.
      raise ValueError('Either observed or occluded agents must be requested.')

    assert_shapes([
        (agent_filter, [batch_size, num_agents, 1, 1]),
    ])

    agent_valid = tf.logical_and(agent_valid, agent_filter)

  return _SampledPoints(
      x=sampled_points.x,
      y=sampled_points.y,
      z=sampled_points.z,
      agent_type=sampled_points.agent_type,
      valid=agent_valid,
  )


def _sample_agent_points(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    points_per_side_length: int,
    points_per_side_width: int,
    translate_sdc_to_origin: bool,
    normalize_sdc_yaw: bool,
) -> _SampledPoints:
  """Creates a set of points to represent agents in the scene.

  For each timestep in `times`, samples the interior of each agent bounding box
  on a uniform grid to create a set of points representing the agent.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    points_per_side_length: The number of points along the length of the agent.
    points_per_side_width: The number of points along the width of the agent.
    translate_sdc_to_origin: If true, translate the points such that the
      autonomous vehicle is at the origin.
    normalize_sdc_yaw: If true, transform the scene such that the autonomous
      vehicle is heading up at the current time.

  Returns:
    _SampledPoints object.
  """
  if normalize_sdc_yaw and not translate_sdc_to_origin:
    raise ValueError('normalize_sdc_yaw requires translate_sdc_to_origin.')

  # All fields: [batch_size, num_agents, num_steps, 1].
  
  x = _stack_field(inputs, times, 'x')
  y = _stack_field(inputs, times, 'y')
  z = _stack_field(inputs, times, 'z')
  bbox_yaw = _stack_field(inputs, times, 'bbox_yaw')
  length = _stack_field(inputs, times, 'length')
  width = _stack_field(inputs, times, 'width')
  agent_type = _stack_field(inputs, times, 'type')
  valid = _stack_field(inputs, times, 'valid')
  shape = ['batch_size', 'num_agents', 'num_steps', 1]
  tf.debugging.assert_shapes([
      (x, shape),
      (y, shape),
      (z, shape),
      (bbox_yaw, shape),
      (length, shape),
      (width, shape),
      (valid, shape),
  ])

  # Translate all agent coordinates such that the autonomous vehicle is at the
  # origin.
  if translate_sdc_to_origin:
    sdc_x = inputs['sdc/current/x'][:, tf.newaxis, tf.newaxis, :]
    sdc_y = inputs['sdc/current/y'][:, tf.newaxis, tf.newaxis, :]
    sdc_z = inputs['sdc/current/z'][:, tf.newaxis, tf.newaxis, :]
    x = x - sdc_x
    y = y - sdc_y
    z = z - sdc_z

  if normalize_sdc_yaw:
    angle = math.pi / 2 - inputs['sdc/current/bbox_yaw'][:, tf.newaxis,
                                                         tf.newaxis, :]
    x, y = rotate_points_around_origin(x, y, angle)
    bbox_yaw = bbox_yaw + angle

  return _sample_points_from_agent_boxes(
      x=x,
      y=y,
      z=z,
      bbox_yaw=bbox_yaw,
      width=width,
      length=length,
      agent_type=agent_type,
      valid=valid,
      points_per_side_length=points_per_side_length,
      points_per_side_width=points_per_side_width,
  )


def _sample_points_from_agent_boxes(
    x: tf.Tensor,
    y: tf.Tensor,
    z: tf.Tensor,
    bbox_yaw: tf.Tensor,
    width: tf.Tensor,
    length: tf.Tensor,
    agent_type: tf.Tensor,
    valid: tf.Tensor,
    points_per_side_length: int,
    points_per_side_width: int,
) -> _SampledPoints:
  """Create a set of 3D points by sampling the interior of agent boxes.

  For each state in the inputs, a set of points_per_side_length *
  points_per_side_width points are generated by sampling within each box.

  Args:
    x: Centers of agent boxes X [..., 1] (any shape with last dim 1).
    y: Centers of agent boxes Y [..., 1] (same shape as x).
    z: Centers of agent boxes Z [..., 1] (same shape as x).
    bbox_yaw: Agent box orientations [..., 1] (same shape as x).
    width : Widths of agent boxes [..., 1] (same shape as x).
    length: Lengths of agent boxes [..., 1] (same shape as x).
    agent_type: Types of agents [..., 1] (same shape as x).
    valid: Agent state valid flag [..., 1] (same shape as x).
    points_per_side_length: The number of points along the length of the agent.
    points_per_side_width: The number of points along the width of the agent.

  Returns:
    _SampledPoints object.
  """
  assert_shapes = tf.debugging.assert_shapes
  assert_shapes([(x, [..., 1])])
  x_shape = x.get_shape().as_list()
  assert_shapes([(y, x_shape), (z, x_shape), (bbox_yaw, x_shape),
                 (width, x_shape), (length, x_shape), (valid, x_shape)])
  if points_per_side_length < 1:
    raise ValueError('points_per_side_length must be >= 1')
  if points_per_side_width < 1:
    raise ValueError('points_per_side_width must be >= 1')

  # Create sample points on a unit square or boundary depending on flag.
  if points_per_side_length == 1:
    step_x = 0.0
  else:
    step_x = 1.0 / (points_per_side_length - 1)
  if points_per_side_width == 1:
    step_y = 0.0
  else:
    step_y = 1.0 / (points_per_side_width - 1)
  unit_x = []
  unit_y = []
  for xi in range(points_per_side_length):
    for yi in range(points_per_side_width):
      unit_x.append(xi * step_x - 0.5)
      unit_y.append(yi * step_y - 0.5)

  # Center unit_x and unit_y if there was only 1 point on those dimensions.
  if points_per_side_length == 1:
    unit_x = np.array(unit_x) + 0.5
  if points_per_side_width == 1:
    unit_y = np.array(unit_y) + 0.5

  unit_x = tf.convert_to_tensor(unit_x, tf.float32)
  unit_y = tf.convert_to_tensor(unit_y, tf.float32)

  num_points = points_per_side_length * points_per_side_width
  assert_shapes([(unit_x, [num_points]), (unit_y, [num_points])])

  # Transform the unit square points to agent dimensions and coordinate frames.
  sin_yaw = tf.sin(bbox_yaw)
  cos_yaw = tf.cos(bbox_yaw)

  # [..., num_points]
  tx = cos_yaw * length * unit_x - sin_yaw * width * unit_y + x
  ty = sin_yaw * length * unit_x + cos_yaw * width * unit_y + y
  tz = tf.broadcast_to(z, tx.shape)

  points_shape = x_shape[:-1] + [num_points]
  assert_shapes([(tx, points_shape), (ty, points_shape), (tz, points_shape)])
  agent_type = tf.broadcast_to(agent_type, tx.shape)
  valid = tf.broadcast_to(valid, tx.shape)

  return _SampledPoints(x=tx, y=ty, z=tz, agent_type=agent_type, valid=valid)


def rotate_points_around_origin(
    x: tf.Tensor,
    y: tf.Tensor,
    angle: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Rotates points around the origin.

  Args:
    x: Tensor of shape [batch_size, ...].
    y: Tensor of shape [batch_size, ...].
    angle: Tensor of shape [batch_size, ...].

  Returns:
    Rotated x, y, each with shape [batch_size, ...].
  """
  tx = tf.cos(angle) * x - tf.sin(angle) * y
  ty = tf.sin(angle) * x + tf.cos(angle) * y
  return tx, ty


def _stack_field(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    field: str,
) -> tf.Tensor:
  """Stack requested field from all agents over specified time frames.

  NOTE: Always adds a last dimension with size 1.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    field: The field to retrieve.

  Returns:
    A tensor containing the requested field over the requested time frames
    with shape [batch_size, num_agents, num_steps, 1].
  """
  if field == 'type':
    # [batch_size, num_agents]
    fields = inputs['state/type']
    # The `type` field's shape is different from other fields.  Broadcast it
    # to have the same shape as another field.
    x = _stack_field(inputs, times, field='x')
    # [batch_size, num_agents, num_steps, 1]
    fields = tf.broadcast_to(fields[:, :, tf.newaxis, tf.newaxis], x.shape)
  else:
    # [batch_size, num_agents, num_steps]
    fields = tf.concat([inputs[f'state/{t}/{field}'] for t in times], axis=-1)
    # [batch_size, num_agents, num_steps, 1]
    fields = fields[:, :, :, tf.newaxis]
  return fields


def _get_num_steps_from_times(
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig) -> int:
  """Returns number of timesteps that exist in requested times."""
  num_steps = 0
  if 'past' in times:
    num_steps += config.num_past_steps
  if 'current' in times:
    num_steps += 1
  if 'future' in times:
    num_steps += config.num_future_steps
  return num_steps


def add_sdc_fields(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Extracts current x, y, z of the autonomous vehicle as specific fields."""
    sdc_indices = np.argwhere(inputs['state/is_sdc'] == True)
    sdc_indices = np.squeeze(sdc_indices, 0)
    # print(sdc_indices)
    inputs['sdc/current/x'] = tf.convert_to_tensor(np.expand_dims(inputs['state/current/x'][sdc_indices[0], sdc_indices[1]], 0), tf.float32)
    inputs['sdc/current/y'] = tf.convert_to_tensor(np.expand_dims(inputs['state/current/y'][sdc_indices[0], sdc_indices[1]], 0), tf.float32)
    inputs['sdc/current/z'] = tf.convert_to_tensor(np.expand_dims(inputs['state/current/z'][sdc_indices[0], sdc_indices[1]], 0), tf.float32)
    inputs['sdc/current/bbox_yaw'] = tf.convert_to_tensor(np.expand_dims(inputs['state/current/bbox_yaw'][sdc_indices[0], sdc_indices[1]], 0), tf.float32)
    return inputs
  
  
def read_data_proto(proto):
  inputs = {}
  inputs['scenario/id'] = proto.scenario_id
  
  inputs['roadgraph_samples/type'] = []
  # inputs['roadgraph_samples/valid'] = []
  inputs['roadgraph_samples/xyz'] = []
  for map_feature in proto.map_features:
      if map_feature.HasField('crosswalk'):
          for polygon in map_feature.crosswalk.polygon:
              inputs['roadgraph_samples/type'].append([0])
              inputs['roadgraph_samples/xyz'].append([polygon.x, polygon.y, polygon.z])
              
      if map_feature.HasField('speed_bump'):
          for polygon in map_feature.speed_bump.polygon:
              inputs['roadgraph_samples/type'].append([1])
              inputs['roadgraph_samples/xyz'].append([polygon.x, polygon.y, polygon.z])
              
      if map_feature.HasField('stop_sign'):
          inputs['roadgraph_samples/type'].append([2])
          inputs['roadgraph_samples/xyz'].append([map_feature.stop_sign.position.x, map_feature.stop_sign.position.y, map_feature.stop_sign.position.z])
          
      if map_feature.HasField('lane'):
          for polyline in map_feature.lane.polyline:
              inputs['roadgraph_samples/type'].append([map_feature.lane.type + 3])
              inputs['roadgraph_samples/xyz'].append([polyline.x, polyline.y, polyline.z])
              
      if map_feature.HasField('road_edge'):
          if len(map_feature.road_edge.polyline) > 2:
              for polyline in map_feature.road_edge.polyline:
                  inputs['roadgraph_samples/type'].append([map_feature.road_edge.type + 3 + 4])
                  inputs['roadgraph_samples/xyz'].append([polyline.x, polyline.y, polyline.z])
              
      if map_feature.HasField('road_line'):
          if len(map_feature.road_line.polyline) > 2:
              for polyline in map_feature.road_line.polyline:
                  inputs['roadgraph_samples/type'].append([map_feature.road_line.type + 3 + 4 + 3])
                  inputs['roadgraph_samples/xyz'].append([polyline.x, polyline.y, polyline.z])
                
  inputs['state/id'] = []
  inputs['state/type'] = []
  inputs['state/is_sdc'] = []
  inputs['state/tracks_to_predict'] = []
  for pred in proto.tracks_to_predict:
      inputs['state/tracks_to_predict'].append([pred.track_index])
      
  for tense in ['past', 'current', 'future']:
      inputs['state/{}/bbox_yaw'.format(tense)] = []
      inputs['state/{}/height'.format(tense)] = []
      inputs['state/{}/width'.format(tense)] = []
      inputs['state/{}/length'.format(tense)] = []
      inputs['state/{}/timestamp_micros'.format(tense)] = []
      inputs['state/{}/valid'.format(tense)] = []
      inputs['state/{}/vel_yaw'.format(tense)] = []
      inputs['state/{}/velocity_x'.format(tense)] = []
      inputs['state/{}/velocity_y'.format(tense)] = []
      inputs['state/{}/x'.format(tense)] = []
      inputs['state/{}/y'.format(tense)] = []
      inputs['state/{}/z'.format(tense)] = []
      
  for idx, track in enumerate(proto.tracks):
      inputs['state/id'].append(track.id)
      inputs['state/type'].append(track.object_type)
      inputs['state/is_sdc'].append(proto.sdc_track_index == idx)
      
      for tense in ['past', 'current', 'future']:
          if tense == 'past':
              start_time = 0
              end_time = 10
          if tense == 'current':
              start_time = 10
              end_time = 11
          if tense == 'future':
              start_time = 11
              end_time = 91
          
          time_span = end_time - start_time
          
          # valid_temp = [-1] * time_span
          valid_temp = []
          bbox_yaw_temp = [-1] * time_span
          height_temp = [-1] * time_span
          width_temp = [-1] * time_span
          length_temp = [-1] * time_span
          timestamp_temp = [-1] * time_span
          vel_yaw_temp = [-1] * time_span
          vel_x_temp = [-1] * time_span
          vel_y_temp = [-1] * time_span
          x_temp = [-1] * time_span
          y_temp = [-1] * time_span
          z_temp = [-1] * time_span
          
          for idx, timestep in enumerate(range(start_time, end_time)):
              state_ = track.states[timestep]
              is_valid = state_.valid
              valid_temp.append(is_valid)
              
              if is_valid:
                  # print(timestep)
                  bbox_yaw_temp[idx] = state_.heading
                  height_temp[idx] = state_.height
                  width_temp[idx] = state_.width
                  length_temp[idx] = state_.length
                  timestamp_temp[idx] = proto.timestamps_seconds[timestep]
                  if state_.velocity_x != 0:
                      vel_yaw_temp[idx] = np.arctan(state_.velocity_y / state_.velocity_x)
                  vel_x_temp[idx] = state_.velocity_x
                  vel_y_temp[idx] = state_.velocity_y 
                  x_temp[idx] = state_.center_x
                  y_temp[idx] = state_.center_y
                  z_temp[idx] = state_.center_z
                  
          inputs['state/{}/bbox_yaw'.format(tense)].append(bbox_yaw_temp)
          inputs['state/{}/height'.format(tense)].append(height_temp)
          inputs['state/{}/width'.format(tense)].append(width_temp)
          inputs['state/{}/length'.format(tense)].append(length_temp)
          inputs['state/{}/timestamp_micros'.format(tense)].append(timestamp_temp)
          inputs['state/{}/valid'.format(tense)].append(valid_temp)
          inputs['state/{}/vel_yaw'.format(tense)].append(vel_yaw_temp)
          inputs['state/{}/velocity_x'.format(tense)].append(vel_x_temp)
          inputs['state/{}/velocity_y'.format(tense)].append(vel_y_temp)
          inputs['state/{}/x'.format(tense)].append(x_temp)
          inputs['state/{}/y'.format(tense)].append(y_temp)
          inputs['state/{}/z'.format(tense)].append(z_temp)
  
  for tense in ['past', 'current']:
    inputs['traffic_light_state/{}/state'.format(tense)] = []
    inputs['traffic_light_state/{}/x'.format(tense)] = []
    inputs['traffic_light_state/{}/y'.format(tense)] = []
    inputs['traffic_light_state/{}/z'.format(tense)] = []
    
  for tense in ['past', 'current']:
      if tense == 'past':
          start_time = 0
          end_time = 10
      if tense == 'current':
          start_time = 10
          end_time = 11
          
      time_span = end_time - start_time
      state_temp = [-1] * time_span
      x_temp = [-1] * time_span
      y_temp = [-1] * time_span
      z_temp = [-1] * time_span
      
      for idx, timestep in enumerate(range(start_time, end_time)):
          state_temp = []
          x_temp = []
          y_temp = []
          z_temp = []
          
          for light in proto.dynamic_map_states[timestep].lane_states:
              state_temp.append(light.state)
              x_temp.append(light.stop_point.x)
              y_temp.append(light.stop_point.y)
              z_temp.append(light.stop_point.z)
          
          inputs['traffic_light_state/{}/state'.format(tense)].append(state_temp)
          inputs['traffic_light_state/{}/x'.format(tense)].append(x_temp)
          inputs['traffic_light_state/{}/y'.format(tense)].append(y_temp)
          inputs['traffic_light_state/{}/z'.format(tense)].append(z_temp)
          
  return inputs


def create_figure_and_axes(size_pixels):
  """Initializes a unique figure and axes for plotting."""
  fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

  # Sets output image to pixel resolution.
  dpi = 100
  size_inches = size_pixels / dpi
  fig.set_size_inches([size_inches, size_inches])
  fig.set_dpi(dpi)
  fig.set_facecolor('white')
  ax.set_facecolor('white')
  ax.xaxis.label.set_color('black')
  ax.tick_params(axis='x', colors='black')
  ax.yaxis.label.set_color('black')
  ax.tick_params(axis='y', colors='black')
  fig.set_tight_layout(True)
  ax.grid(False)
  return fig, ax


def fig_canvas_image(fig):
  """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
  # Just enough margin in the figure to display xticks and yticks.
  fig.subplots_adjust(
      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
  """Compute a color map array of shape [num_agents, 4]."""
  colors = plt.cm.get_cmap('jet', num_agents)
  colors = colors(range(num_agents))
  np.random.shuffle(colors)
  return colors


def get_viewport(all_states, all_states_mask):
  """Gets the region containing the data.

  Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,
      2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for
      `all_states`.

  Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
  """
  valid_states = all_states[all_states_mask]
  all_y = valid_states[..., 1]
  all_x = valid_states[..., 0]

  center_y = (np.max(all_y) + np.min(all_y)) / 2
  center_x = (np.max(all_x) + np.min(all_x)) / 2

  range_y = np.ptp(all_y)
  range_x = np.ptp(all_x)

  width = max(range_y, range_x)

  return center_y, center_x, width


def visualize_one_step(
    states,
    mask,
    roadgraph,
    title,
    center_y,
    center_x,
    width,
    color_map,
    size_pixels=1000,
):
  """Generate visualization for a single step."""

  # Create figure and axes.
  fig, ax = create_figure_and_axes(size_pixels=size_pixels)

  # Plot roadgraph.
  rg_pts = roadgraph[:, :2].T
  ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

  masked_x = states[:, 0][mask]
  masked_y = states[:, 1][mask]
  colors = color_map[mask]

  # Plot agent current position.
  ax.scatter(
      masked_x,
      masked_y,
      marker='o',
      linewidths=3,
      color=colors,
  )

  # Title.
  ax.set_title(title)

  # Set axes.  Should be at least 10m on a side.
  size = max(10, width * 1.0)
  ax.axis([
      -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
      size / 2 + center_y
  ])
  ax.set_aspect('equal')

  image = fig_canvas_image(fig)
  plt.close(fig)
  return image


def visualize_all_agents_smooth(
    decoded_example,
    size_pixels=1000,
):
    """Visualizes all agent predicted trajectories in a serie of images.

    Args:
      decoded_example: Dictionary containing agent info about all modeled agents.
      size_pixels: The size in pixels of the output image.

    Returns:
      T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
    """
    past_states = np.stack([decoded_example['state/past/x'], decoded_example['state/past/y']], -1)
    past_states_mask = decoded_example['state/past/valid'] > 0.0

    current_states = np.stack([decoded_example['state/current/x'], decoded_example['state/current/y']], -1)
    current_states_mask = decoded_example['state/current/valid'] > 0.0
    
    future_states = np.stack([decoded_example['state/future/x'], decoded_example['state/future/y']], -1)
    future_states_mask = decoded_example['state/future/valid'] > 0.0
    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz']

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    color_map = get_colormap(num_agents)

    # [num_agents, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agents, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)

    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []

    # Generate images from past time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, 1),
            np.split(past_states_mask, num_past_steps, 1))):
      im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                              'past: %d' % (num_past_steps - i), center_y,
                              center_x, width, color_map, size_pixels)
      images.append(im)

    # Generate one image for the current time step.
    s = current_states
    m = current_states_mask

    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                            center_x, width, color_map, size_pixels)
    images.append(im)

    # Generate images from future time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_states_mask, num_future_steps, 1))):
      im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                              'future: %d' % (i + 1), center_y, center_x, width,
                              color_map, size_pixels)
      images.append(im)

    return images

def frame2video(im_dir, video_dir, fps):
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # opencv版本是3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)

    videoWriter.release()
    print('Done')


def generate_vid_from_pic(im_dir, video_name, fps):
    dir_list=os.listdir(im_dir)
    video_dir = im_dir + video_name + '.avi'
    if os.path.exists(video_dir):
      os.remove(video_dir)
    frame2video(im_dir, video_dir, fps)
    

def occupancy_rgb_image(
    agent_grids: AgentGrids,
    roadgraph_image: tf.Tensor,
) -> tf.Tensor:
  """Visualize predictions or ground-truth occupancy.

  Args:
    agent_grids: AgentGrids object containing optional
      vehicles/pedestrians/cyclists.
    roadgraph_image: Road graph image [batch_size, height, width, 1] float32.

  Returns:
    [batch_size, height, width, 3] float32 RGB image.
  """
  zeros = tf.zeros_like(roadgraph_image)
  ones = tf.ones_like(zeros)

  agents = agent_grids
  veh = zeros if agents.vehicles is None else agents.vehicles
  ped = zeros if agents.pedestrians is None else agents.pedestrians
  cyc = zeros if agents.cyclists is None else agents.cyclists

  # Convert layers to RGB.
  rg_rgb = tf.concat([zeros, zeros, zeros], axis=-1)
  veh_rgb = tf.concat([veh, zeros, zeros], axis=-1)  # Red.
  ped_rgb = tf.concat([zeros, ped * 0.67, zeros], axis=-1)  # Green.
  cyc_rgb = tf.concat([cyc * 0.33, zeros, zeros * 0.33], axis=-1)  # Purple.
  bg_rgb = tf.concat([ones, ones, ones], axis=-1)  # White background.
  # Set alpha layers over all RGB channels.
  rg_a = tf.concat([roadgraph_image, roadgraph_image, roadgraph_image], axis=-1)
  veh_a = tf.concat([veh, veh, veh], axis=-1)
  ped_a = tf.concat([ped, ped, ped], axis=-1)
  cyc_a = tf.concat([cyc, cyc, cyc], axis=-1)
  # Stack layers one by one.
  img, img_a = _alpha_blend(fg=rg_rgb, bg=bg_rgb, fg_a=rg_a)
  img, img_a = _alpha_blend(fg=veh_rgb, bg=img, fg_a=veh_a, bg_a=img_a)
  img, img_a = _alpha_blend(fg=ped_rgb, bg=img, fg_a=ped_a, bg_a=img_a)
  img, img_a = _alpha_blend(fg=cyc_rgb, bg=img, fg_a=cyc_a, bg_a=img_a)
  return img


def flow_rgb_image(
    flow: tf.Tensor,
    roadgraph_image: tf.Tensor,
    agent_trails: tf.Tensor,
) -> tf.Tensor:
  """Converts (dx, dy) flow to RGB image.

  Args:
    flow: [batch_size, height, width, 2] float32 tensor holding (dx, dy) values.
    roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
    agent_trails: [batch_size, height, width, 1] float32 tensor containing
      rendered trails for all agents over the past and current time frames.

  Returns:
    [batch_size, height, width, 3] float32 RGB image.
  """
  # Swap x, y for compatibilty with published visualizations.
  flow = tf.roll(flow, shift=1, axis=-1)
  # saturate_magnitude=-1 normalizes highest intensity to largest magnitude.
  flow_image = _optical_flow_to_rgb(flow, saturate_magnitude=-1)
  # Add roadgraph.
  flow_image = _add_grayscale_layer(roadgraph_image, flow_image)  # Black.
  # Overlay agent trails.
  flow_image = _add_grayscale_layer(agent_trails * 0.2, flow_image)  # 0.2 alpha
  return flow_image

def flow_rgb_image_per_step(
    flow: tf.Tensor,
) -> tf.Tensor:

  flow = tf.roll(flow, shift=1, axis=-1)
  # saturate_magnitude=-1 normalizes highest intensity to largest magnitude.
  flow_image = _optical_flow_to_rgb(flow, saturate_magnitude=-1)

  return flow_image


def _add_grayscale_layer(
    fg_a: tf.Tensor,
    scene_rgb: tf.Tensor,
) -> tf.Tensor:
  """Adds a black/gray layer using fg_a as alpha over an RGB image."""
  # Create a black layer matching dimensions of fg_a.
  black = tf.zeros_like(fg_a)
  black = tf.concat([black, black, black], axis=-1)
  # Add the black layer with transparency over the scene_rgb image.
  overlay, _ = _alpha_blend(fg=black, bg=scene_rgb, fg_a=fg_a, bg_a=1.0)
  return overlay


def _alpha_blend(
    fg: tf.Tensor,
    bg: tf.Tensor,
    fg_a: Optional[tf.Tensor] = None,
    bg_a: Optional[Union[tf.Tensor, float]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Overlays foreground and background image with custom alpha values.

  Implements alpha compositing using Porter/Duff equations.
  https://en.wikipedia.org/wiki/Alpha_compositing

  Works with 1-channel or 3-channel images.

  If alpha values are not specified, they are set to the intensity of RGB
  values.

  Args:
    fg: Foreground: float32 tensor shaped [batch, grid_height, grid_width, d].
    bg: Background: float32 tensor shaped [batch, grid_height, grid_width, d].
    fg_a: Foreground alpha: float32 tensor broadcastable to fg.
    bg_a: Background alpha: float32 tensor broadcastable to bg.

  Returns:
    Output image: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
    Output alpha: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
  """
  if fg_a is None:
    fg_a = fg
  if bg_a is None:
    bg_a = bg
  eps = tf.keras.backend.epsilon()
  out_a = fg_a + bg_a * (1 - fg_a)
  out_rgb = (fg * fg_a + bg * bg_a * (1 - fg_a)) / (out_a + eps)
  return out_rgb, out_a



def _optical_flow_to_hsv(
    flow: tf.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Visualize an optical flow field in HSV colorspace.

  This uses the standard color code with hue corresponding to direction of
  motion and saturation corresponding to magnitude.

  The attr `saturate_magnitude` sets the magnitude of motion (in pixels) at
  which the color code saturates. A negative value is replaced with the maximum
  magnitude in the optical flow field.

  Args:
    flow: A `Tensor` of type `float32`. A 3-D or 4-D tensor storing (a batch of)
      optical flow field(s) as flow([batch,] i, j) = (dx, dy). The shape of the
      tensor is [height, width, 2] or [batch, height, width, 2] for the 4-D
      case.
    saturate_magnitude: An optional `float`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    An tf.float32 HSV image (or image batch) of size [height, width, 3]
    (or [batch, height, width, 3]) compatible with tensorflow color conversion
    ops. The hue at each pixel corresponds to direction of motion. The
    saturation at each pixel corresponds to the magnitude of motion relative to
    the `saturate_magnitude` value. Hue, saturation, and value are in [0, 1].
  """
  with tf.name_scope(name or 'OpticalFlowToHSV'):
    flow_shape = flow.shape
    if len(flow_shape) < 3:
      raise ValueError('flow must be at least 3-dimensional, got'
                       f' `{flow_shape}`')
    if flow_shape[-1] != 2:
      raise ValueError(f'flow must have innermost dimension of 2, got'
                       f' `{flow_shape}`')
    height = flow_shape[-3]
    width = flow_shape[-2]
    flow_flat = tf.reshape(flow, (-1, height, width, 2))

    dx = flow_flat[..., 0]
    dy = flow_flat[..., 1]
    # [batch_size, height, width]
    magnitudes = tf.sqrt(tf.square(dx) + tf.square(dy))
    if saturate_magnitude < 0:
      # [batch_size, 1, 1]
      local_saturate_magnitude = tf.reduce_max(
          magnitudes, axis=(1, 2), keepdims=True)
    else:
      local_saturate_magnitude = tf.convert_to_tensor(saturate_magnitude)

    # Hue is angle scaled to [0.0, 1.0).
    hue = (tf.math.mod(tf.math.atan2(dy, dx), (2 * math.pi))) / (2 * math.pi)
    # Saturation is relative magnitude.
    relative_magnitudes = tf.math.divide_no_nan(magnitudes,
                                                local_saturate_magnitude)
    saturation = tf.minimum(
        relative_magnitudes,
        1.0  # Larger magnitudes saturate.
    )
    # Value is fixed.
    value = tf.ones_like(saturation)
    hsv_flat = tf.stack((hue, saturation, value), axis=-1)
    return tf.reshape(hsv_flat, flow_shape.as_list()[:-1] + [3])


def _optical_flow_to_rgb(
    flow: tf.Tensor,
    saturate_magnitude: float = -1.0,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Visualize an optical flow field in RGB colorspace."""
  name = name or 'OpticalFlowToRGB'
  hsv = _optical_flow_to_hsv(flow, saturate_magnitude, name)
  return tf.image.hsv_to_rgb(hsv)

def numpy_to_tf(inputs_batch):
  inputs_batch['scenario/id'] = tf.convert_to_tensor(inputs_batch['scenario/id'], tf.string)
  inputs_batch['roadgraph_samples/type'] = tf.convert_to_tensor(inputs_batch['roadgraph_samples/type'], tf.int64)
  inputs_batch['roadgraph_samples/xyz'] = tf.convert_to_tensor(inputs_batch['roadgraph_samples/xyz'], tf.float32)

  inputs_batch['state/id'] = tf.convert_to_tensor(inputs_batch['state/id'], tf.float32)
  inputs_batch['state/type'] = tf.convert_to_tensor(inputs_batch['state/type'], tf.int64)
  inputs_batch['state/is_sdc'] = tf.convert_to_tensor(inputs_batch['state/is_sdc'], tf.int64)
  inputs_batch['state/tracks_to_predict'] = tf.convert_to_tensor(inputs_batch['state/tracks_to_predict'], tf.int64)

  for tense in ['past', 'current', 'future']:
      for attr in ['timestamp_micros', 'valid']:
          inputs_batch['state/{}/{}'.format(tense, attr)] = tf.convert_to_tensor(inputs_batch['state/{}/{}'.format(tense, attr)], tf.int64)
          
      for attr in ['width', 'height', 'length', 'bbox_yaw', 'vel_yaw', 'velocity_x', 'velocity_y', 'x', 'y', 'z']:
          inputs_batch['state/{}/{}'.format(tense, attr)] = tf.convert_to_tensor(inputs_batch['state/{}/{}'.format(tense, attr)], tf.float32)
          
  # for tense in ['past', 'current']:
  #     inputs_batch['traffic_light_state/{}/state'.format(tense)] = tf.convert_to_tensor(inputs_batch['traffic_light_state/{}/state'.format(tense)], tf.int64)
  #     for attr in ['x', 'y', 'z']:
  #         inputs_batch['traffic_light_state/{}/{}'.format(tense, attr)] = tf.convert_to_tensor(inputs_batch['traffic_light_state/{}/{}'.format(tense, attr)], tf.float32)
          
  return inputs_batch


def generate_img_from_occupancy_grid(grid, color = 'r'):
  img = deepcopy(grid)
  idx = np.where(np.array(img))
  img = img[..., tf.newaxis]
  ones = tf.ones_like(img)
  fig = np.array(tf.concat([ones, ones, ones], axis=-1))*255
  if color == 'r':
    fig[idx] = np.array([0, 0, 255])
  if color == 'b':
    fig[idx] = np.array([0, 0, 255])
  
  return fig