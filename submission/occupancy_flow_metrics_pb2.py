# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: occupancy_flow_metrics.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='occupancy_flow_metrics.proto',
  package='waymo.open_dataset',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1coccupancy_flow_metrics.proto\x12\x12waymo.open_dataset\"\x9b\x03\n\x17OccupancyFlowTaskConfig\x12\x1a\n\x0enum_past_steps\x18\x01 \x01(\x05:\x02\x31\x30\x12\x1c\n\x10num_future_steps\x18\x02 \x01(\x05:\x02\x38\x30\x12\x18\n\rnum_waypoints\x18\x03 \x01(\x05:\x01\x38\x12\"\n\x14\x63umulative_waypoints\x18\x04 \x01(\x08:\x04true\x12\x1f\n\x11normalize_sdc_yaw\x18\x0c \x01(\x08:\x04true\x12\x1e\n\x11grid_height_cells\x18\x05 \x01(\x05:\x03\x32\x35\x36\x12\x1d\n\x10grid_width_cells\x18\x06 \x01(\x05:\x03\x32\x35\x36\x12\x1a\n\rsdc_y_in_grid\x18\x07 \x01(\x05:\x03\x31\x39\x32\x12\x1a\n\rsdc_x_in_grid\x18\x08 \x01(\x05:\x03\x31\x32\x38\x12\x1d\n\x10pixels_per_meter\x18\t \x01(\x02:\x03\x33.2\x12(\n\x1c\x61gent_points_per_side_length\x18\n \x01(\x05:\x02\x34\x38\x12\'\n\x1b\x61gent_points_per_side_width\x18\x0b \x01(\x05:\x02\x31\x36\"\x85\x02\n\x14OccupancyFlowMetrics\x12\x1d\n\x15vehicles_observed_auc\x18\x01 \x01(\x02\x12\x1d\n\x15vehicles_observed_iou\x18\x02 \x01(\x02\x12\x1d\n\x15vehicles_occluded_auc\x18\x03 \x01(\x02\x12\x1d\n\x15vehicles_occluded_iou\x18\x04 \x01(\x02\x12\x19\n\x11vehicles_flow_epe\x18\x05 \x01(\x02\x12*\n\"vehicles_flow_warped_occupancy_auc\x18\x06 \x01(\x02\x12*\n\"vehicles_flow_warped_occupancy_iou\x18\x07 \x01(\x02'
)




_OCCUPANCYFLOWTASKCONFIG = _descriptor.Descriptor(
  name='OccupancyFlowTaskConfig',
  full_name='waymo.open_dataset.OccupancyFlowTaskConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_past_steps', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.num_past_steps', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_future_steps', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.num_future_steps', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=80,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_waypoints', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.num_waypoints', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cumulative_waypoints', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.cumulative_waypoints', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='normalize_sdc_yaw', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.normalize_sdc_yaw', index=4,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='grid_height_cells', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.grid_height_cells', index=5,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='grid_width_cells', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.grid_width_cells', index=6,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sdc_y_in_grid', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.sdc_y_in_grid', index=7,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=192,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sdc_x_in_grid', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.sdc_x_in_grid', index=8,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=128,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pixels_per_meter', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.pixels_per_meter', index=9,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(3.2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='agent_points_per_side_length', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.agent_points_per_side_length', index=10,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=48,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='agent_points_per_side_width', full_name='waymo.open_dataset.OccupancyFlowTaskConfig.agent_points_per_side_width', index=11,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=53,
  serialized_end=464,
)


_OCCUPANCYFLOWMETRICS = _descriptor.Descriptor(
  name='OccupancyFlowMetrics',
  full_name='waymo.open_dataset.OccupancyFlowMetrics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='vehicles_observed_auc', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_observed_auc', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vehicles_observed_iou', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_observed_iou', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vehicles_occluded_auc', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_occluded_auc', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vehicles_occluded_iou', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_occluded_iou', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vehicles_flow_epe', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_flow_epe', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vehicles_flow_warped_occupancy_auc', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_flow_warped_occupancy_auc', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vehicles_flow_warped_occupancy_iou', full_name='waymo.open_dataset.OccupancyFlowMetrics.vehicles_flow_warped_occupancy_iou', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=467,
  serialized_end=728,
)

DESCRIPTOR.message_types_by_name['OccupancyFlowTaskConfig'] = _OCCUPANCYFLOWTASKCONFIG
DESCRIPTOR.message_types_by_name['OccupancyFlowMetrics'] = _OCCUPANCYFLOWMETRICS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OccupancyFlowTaskConfig = _reflection.GeneratedProtocolMessageType('OccupancyFlowTaskConfig', (_message.Message,), {
  'DESCRIPTOR' : _OCCUPANCYFLOWTASKCONFIG,
  '__module__' : 'occupancy_flow_metrics_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.OccupancyFlowTaskConfig)
  })
_sym_db.RegisterMessage(OccupancyFlowTaskConfig)

OccupancyFlowMetrics = _reflection.GeneratedProtocolMessageType('OccupancyFlowMetrics', (_message.Message,), {
  'DESCRIPTOR' : _OCCUPANCYFLOWMETRICS,
  '__module__' : 'occupancy_flow_metrics_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.OccupancyFlowMetrics)
  })
_sym_db.RegisterMessage(OccupancyFlowMetrics)


# @@protoc_insertion_point(module_scope)
