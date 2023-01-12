CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
--exp_name=test_5 \
--num_workers=8 \
--epoch_num=1000 \
--learning_rate=1e-3 \
--print_freq=25 \

--val_preprocessed_file_path='/GPFS/rhome/ziwang/projects/occupancy_flow/dataset/train_preprocessed_data'