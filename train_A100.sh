CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
--num_workers=16 \
--exp_name='test_424' \
--use_occluded_loss \
--use_flow_loss \
--train_preprocessed_file_path='/nvme/jxs/WaymoOpenMotion/scenario/occupancy_flow/train_preprocessed_data' \
--val_preprocessed_file_path='/nvme/jxs/WaymoOpenMotion/scenario/occupancy_flow/val_preprocessed_data' \
--batch_size=20 \
--print_freq=100