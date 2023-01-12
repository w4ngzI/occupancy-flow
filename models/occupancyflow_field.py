import nntplib
import torch
import torch.nn as nn
from .pointpillar import *
from .EfficientDet.backbone import EfficientDetBackbone
import numpy as np
from models.resnet import *

class OccupancyFlow_Field(nn.Module):
    def __init__(self, args):
        super(OccupancyFlow_Field, self).__init__()
        self.args = args
        # self.points_features_dim = 3 + args.embedding_dim_type + args.embedding_dim_time
        self.points_features_dim = 5
        self.voxel_size = torch.tensor([1/3.2, 1/3.2])
        self.point_cloud_range = torch.tensor([-40, -20, 40, 60])  #[x_min, y_min, x_max, y_max]
        
        self.point_pillar_agent = PillarVFE(self.points_features_dim, self.voxel_size, self.point_cloud_range, self.args.pillar_encoding_dim_agent, False)
        self.type_embedding = nn.Embedding(args.type_num + 1, args.embedding_dim_type, padding_idx = 0)
        self.time_embedding = nn.Embedding(args.time_num + 1, args.embedding_dim_time, padding_idx = 0)
        self.pillar_scatter_agent = PointPillarScatter(args, args.pillar_encoding_dim_agent)
        
        self.point_pillar_rg = PillarVFE(3, self.voxel_size, self.point_cloud_range, self.args.pillar_encoding_dim_rg, False)
        self.pillar_scatter_rg = PointPillarScatter(args, args.pillar_encoding_dim_rg)
        
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.output_channels = self.fpn_num_filters[self.args.compound_coef]
        self.output_size = 64
        self.upsample_before_decoder_channels = self.args.pillar_encoding_dim_agent + self.args.pillar_encoding_dim_rg
        self.upsample_before_decoder = nn.ConvTranspose2d(self.upsample_before_decoder_channels, self.upsample_before_decoder_channels, 3, stride=2, padding=1)
        self.decoder = EfficientDetBackbone(num_classes=3, compound_coef=self.args.compound_coef)
        self.upsample_after_decoder_1 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 3, stride = 2, padding = 1)
        self.batch_norm_1 = nn.BatchNorm2d(num_features = self.output_channels)
        self.upsample_after_decoder_2 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 3, stride = 2, padding = 1)
        self.batch_norm_2 = nn.BatchNorm2d(num_features = self.output_channels)
        # self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        self.mish = nn.Mish()
        self.conv = nn.Conv2d(self.output_channels, self.args.pred_channels_num, 3, stride=1, padding=1)
        

    def forward(self, agent_points_features, rg_points_features):
        
        voxels_agent, coors_agent, num_points_per_voxel_agent = agent_points_features['voxels'].cuda(), agent_points_features['coors'].cuda(), agent_points_features['num_points_per_voxel'].cuda()
        xyz = voxels_agent[..., :3]
        type_idx = voxels_agent[..., -2].long()
        time_idx = voxels_agent[..., -1].long()
        # print('time max', torch.max(time_idx))
        # print('time min', torch.min(time_idx))
        # print('type max', torch.max(type_idx))
        # print('type min', torch.min(type_idx))
        # print('voxels_agent in model', torch.isnan(voxels_agent).int().sum())
        # type_concat = self.type_embedding(type_idx)
        # time_concat = self.time_embedding(time_idx)
        # voxels_agent = torch.cat((xyz, type_concat, time_concat), -1).cuda()
        
        pillar_features_agent = self.point_pillar_agent(voxels_agent, coors_agent, num_points_per_voxel_agent)
        # print('pillar_features_agent', torch.isnan(pillar_features_agent).int().sum())
        batch_spatial_features_agent = self.pillar_scatter_agent(pillar_features_agent, coors_agent)  #[B, 256, 256, D]  D=64
        # print('batch_spatial_features_agent', torch.isnan(batch_spatial_features_agent).int().sum())
        
        
        voxels_rg, coors_rg, num_points_per_voxel_rg = rg_points_features['voxels'].cuda(), rg_points_features['coors'].cuda(), rg_points_features['num_points_per_voxel'].cuda()
        pillar_features_rg = self.point_pillar_rg(voxels_rg, coors_rg, num_points_per_voxel_rg)
        # print('pillar_features_rg', torch.isnan(pillar_features_rg).int().sum())
        batch_spatial_features_rg = self.pillar_scatter_rg(pillar_features_rg, coors_rg)
        # print('batch_spatial_features_rg', torch.isnan(batch_spatial_features_rg).int().sum())
        batch_spatial_features = torch.cat((batch_spatial_features_agent, batch_spatial_features_rg), -1)
        batch_spatial_features = batch_spatial_features.permute(0, 3, 1, 2).contiguous()     #[B, D, 256, 256]
        # print('batch_spatial_features', batch_spatial_features.shape)
        # np.save('/GPFS/rhome/ziwang/projects/occupancy_flow/vis/vis_files/batch_spatial_features.npy', batch_spatial_features.detach().cpu().numpy())
        output = self.upsample_before_decoder(batch_spatial_features, output_size = (self.args.batch_size, self.upsample_before_decoder_channels, self.args.grid_height_cells*2, self.args.grid_width_cells*2)) #[B, D, 512, 512]
        # print('output 1', torch.isnan(output).int().sum())
        output = self.decoder(output)[0]    #[B, 288, 64, 64]
        # print('output 2', torch.isnan(output).int().sum())
        output = self.upsample_after_decoder_1(output, output_size = (self.args.batch_size, self.output_channels, self.output_size*2, self.output_size*2)) #[B, 288, 128, 128]
        # output = self.mish(output)
        # output = self.batch_norm_1(output)
        output = self.upsample_after_decoder_2(output, output_size = (self.args.batch_size, self.output_channels, self.args.grid_height_cells, self.args.grid_width_cells))
        # output = self.mish(output)
        # output = self.batch_norm_2(output) #[B, 288, 256, 256]
        output = self.conv(output)   #[B, 32, 256, 256]
        # print('output 3', torch.isnan(output).int().sum())
        # print('output', output.shape)
        return output.permute(0, 2, 3, 1).contiguous()

class OccupancyFlow_Field_v2(nn.Module):
    def __init__(self, args):
        super(OccupancyFlow_Field_v2, self).__init__()
        self.args = args
        # self.points_features_dim = 3 + args.embedding_dim_type + args.embedding_dim_time
        self.points_features_dim = 5
        self.voxel_size = torch.tensor([1/3.2, 1/3.2])
        self.point_cloud_range = torch.tensor([-40, -20, 40, 60])  #[x_min, y_min, x_max, y_max]
        
        self.point_pillar_agent = PillarVFE(self.points_features_dim, self.voxel_size, self.point_cloud_range, self.args.pillar_encoding_dim_agent, False)
        self.type_embedding = nn.Embedding(args.type_num + 1, args.embedding_dim_type, padding_idx = 0)
        self.time_embedding = nn.Embedding(args.time_num + 1, args.embedding_dim_time, padding_idx = 0)
        self.pillar_scatter_agent = PointPillarScatter(args, args.pillar_encoding_dim_agent)
        
        self.point_pillar_rg = PillarVFE(3, self.voxel_size, self.point_cloud_range, self.args.pillar_encoding_dim_rg, False)
        self.pillar_scatter_rg = PointPillarScatter(args, args.pillar_encoding_dim_rg)
        
        # self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        # self.output_channels = self.fpn_num_filters[self.args.compound_coef]
        # self.output_size = 64
        # self.upsample_before_decoder_channels = self.args.pillar_encoding_dim_agent + self.args.pillar_encoding_dim_rg
        # self.upsample_before_decoder = nn.ConvTranspose2d(self.upsample_before_decoder_channels, self.upsample_before_decoder_channels, 3, stride=2, padding=1)
        # self.decoder = EfficientDetBackbone(num_classes=3, compound_coef=self.args.compound_coef)
        # self.upsample_after_decoder_1 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 3, stride = 2, padding = 1)
        # self.batch_norm_1 = nn.BatchNorm2d(num_features = self.output_channels)
        # self.upsample_after_decoder_2 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 3, stride = 2, padding = 1)
        # self.batch_norm_2 = nn.BatchNorm2d(num_features = self.output_channels)
        # # self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        # # self.mish = nn.Mish()
        # self.conv = nn.Conv2d(self.output_channels, self.args.pred_channels_num, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(args.pillar_encoding_dim_rg + args.pillar_encoding_dim_agent, self.args.pred_channels_num, 3, stride=1, padding=1)
        

    def forward(self, agent_points_features, rg_points_features):
        
        voxels_agent, coors_agent, num_points_per_voxel_agent = agent_points_features['voxels'].cuda(), agent_points_features['coors'].cuda(), agent_points_features['num_points_per_voxel'].cuda()
        xyz = voxels_agent[..., :3]
        type_idx = voxels_agent[..., -2].long()
        time_idx = voxels_agent[..., -1].long()
        # print('time max', torch.max(time_idx))
        # print('time min', torch.min(time_idx))
        # print('type max', torch.max(type_idx))
        # print('type min', torch.min(type_idx))
        # print('voxels_agent in model', torch.isnan(voxels_agent).int().sum())
        # type_concat = self.type_embedding(type_idx)
        # time_concat = self.time_embedding(time_idx)
        # voxels_agent = torch.cat((xyz, type_concat, time_concat), -1).cuda()
        
        pillar_features_agent = self.point_pillar_agent(voxels_agent, coors_agent, num_points_per_voxel_agent)
        # print('pillar_features_agent', torch.isnan(pillar_features_agent).int().sum())
        batch_spatial_features_agent = self.pillar_scatter_agent(pillar_features_agent, coors_agent)  #[B, 256, 256, D]  D=64
        # print('batch_spatial_features_agent', torch.isnan(batch_spatial_features_agent).int().sum())
        
        
        voxels_rg, coors_rg, num_points_per_voxel_rg = rg_points_features['voxels'].cuda(), rg_points_features['coors'].cuda(), rg_points_features['num_points_per_voxel'].cuda()
        pillar_features_rg = self.point_pillar_rg(voxels_rg, coors_rg, num_points_per_voxel_rg)
        # print('pillar_features_rg', torch.isnan(pillar_features_rg).int().sum())
        batch_spatial_features_rg = self.pillar_scatter_agent(pillar_features_rg, coors_rg)
        # print('batch_spatial_features_rg', torch.isnan(batch_spatial_features_rg).int().sum())
        batch_spatial_features = torch.cat((batch_spatial_features_agent, batch_spatial_features_rg), -1)
        batch_spatial_features = batch_spatial_features.permute(0, 3, 1, 2).contiguous()     #[B, D, 256, 256]
        # print('batch_spatial_features', batch_spatial_features.shape)
        # output = self.upsample_before_decoder(batch_spatial_features, output_size = (self.args.batch_size, self.upsample_before_decoder_channels, self.args.grid_height_cells*2, self.args.grid_width_cells*2)) #[B, D, 512, 512]
        # # print('output 1', torch.isnan(output).int().sum())
        # output = self.decoder(output)[0]    #[B, 288, 64, 64]
        # # print('output 2', torch.isnan(output).int().sum())
        # output = self.upsample_after_decoder_1(output, output_size = (self.args.batch_size, self.output_channels, self.output_size*2, self.output_size*2)) #[B, 288, 128, 128]
        # output = self.relu(output)
        # # output = self.batch_norm_1(output)
        # output = self.upsample_after_decoder_2(output, output_size = (self.args.batch_size, self.output_channels, self.args.grid_height_cells, self.args.grid_width_cells))
        # output = self.relu(output)
        # output = self.batch_norm_2(output) #[B, 288, 256, 256]
        output = self.conv(batch_spatial_features)   #[B, 32, 256, 256]
        # print('output 3', torch.isnan(output).int().sum())
        # print('output', output.shape)
        return output.permute(0, 2, 3, 1).contiguous()
    
    
class OccupancyFlow_Field_v3(nn.Module):
    def __init__(self, args):
        super(OccupancyFlow_Field_v3, self).__init__()
        self.args = args
        # self.points_features_dim = 3 + args.embedding_dim_type + args.embedding_dim_time
        self.points_features_dim = 5
        self.voxel_size = torch.tensor([1/3.2, 1/3.2])
        self.point_cloud_range = torch.tensor([-40, -20, 40, 60])  #[x_min, y_min, x_max, y_max]
        
        self.point_pillar_agent = PillarVFE(self.points_features_dim, self.voxel_size, self.point_cloud_range, self.args.pillar_encoding_dim_agent, False)
        self.type_embedding = nn.Embedding(args.type_num + 1, args.embedding_dim_type, padding_idx = 0)
        self.time_embedding = nn.Embedding(args.time_num + 1, args.embedding_dim_time, padding_idx = 0)
        self.pillar_scatter_agent = PointPillarScatter(args, args.pillar_encoding_dim_agent)
        
        self.point_pillar_rg = PillarVFE(3, self.voxel_size, self.point_cloud_range, self.args.pillar_encoding_dim_rg, False)
        self.pillar_scatter_rg = PointPillarScatter(args, args.pillar_encoding_dim_rg)
        
        self.upsample_before_decoder_channels = self.args.pillar_encoding_dim_agent + self.args.pillar_encoding_dim_rg
        self.resnet = resnet50()
        self.output_size = 64
        self.output_channels = 1024
        self.upsample_after_decoder_1 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 3, stride = 2, padding = 1)
        self.upsample_after_decoder_2 = nn.ConvTranspose2d(self.output_channels, self.output_channels, 3, stride = 2, padding = 1)
        # self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU()
        self.mish = nn.Mish()
        self.conv = nn.Conv2d(self.output_channels, self.args.pred_channels_num, 3, stride=1, padding=1)
        

    def forward(self, agent_points_features, rg_points_features):
        
        voxels_agent, coors_agent, num_points_per_voxel_agent = agent_points_features['voxels'].cuda(), agent_points_features['coors'].cuda(), agent_points_features['num_points_per_voxel'].cuda()
        xyz = voxels_agent[..., :3]
        type_idx = voxels_agent[..., -2].long()
        time_idx = voxels_agent[..., -1].long()
        # print('time max', torch.max(time_idx))
        # print('time min', torch.min(time_idx))
        # print('type max', torch.max(type_idx))
        # print('type min', torch.min(type_idx))
        # print('voxels_agent in model', torch.isnan(voxels_agent).int().sum())
        # type_concat = self.type_embedding(type_idx)
        # time_concat = self.time_embedding(time_idx)
        # voxels_agent = torch.cat((xyz, type_concat, time_concat), -1).cuda()
        
        pillar_features_agent = self.point_pillar_agent(voxels_agent, coors_agent, num_points_per_voxel_agent)
        # print('pillar_features_agent', torch.isnan(pillar_features_agent).int().sum())
        batch_spatial_features_agent = self.pillar_scatter_agent(pillar_features_agent, coors_agent)  #[B, 256, 256, D]  D=64
        # print('batch_spatial_features_agent', torch.isnan(batch_spatial_features_agent).int().sum())
        
        
        voxels_rg, coors_rg, num_points_per_voxel_rg = rg_points_features['voxels'].cuda(), rg_points_features['coors'].cuda(), rg_points_features['num_points_per_voxel'].cuda()
        pillar_features_rg = self.point_pillar_rg(voxels_rg, coors_rg, num_points_per_voxel_rg)
        # print('pillar_features_rg', torch.isnan(pillar_features_rg).int().sum())
        batch_spatial_features_rg = self.pillar_scatter_rg(pillar_features_rg, coors_rg)
        # print('batch_spatial_features_rg', torch.isnan(batch_spatial_features_rg).int().sum())
        batch_spatial_features = torch.cat((batch_spatial_features_agent, batch_spatial_features_rg), -1)
        batch_spatial_features = batch_spatial_features.permute(0, 3, 1, 2).contiguous()     #[B, D, 256, 256]
        # print('batch_spatial_features', batch_spatial_features.shape)
        # np.save('/GPFS/rhome/ziwang/projects/occupancy_flow/vis/vis_files/batch_spatial_features.npy', batch_spatial_features.detach().cpu().numpy())
        # output = self.upsample_before_decoder(batch_spatial_features, output_size = (self.args.batch_size, self.upsample_before_decoder_channels, self.args.grid_height_cells*2, self.args.grid_width_cells*2)) #[B, D, 512, 512]
        # # print('output 1', torch.isnan(output).int().sum())
        # output = self.decoder(output)[0]    #[B, 288, 64, 64]
        output = self.resnet(batch_spatial_features)
        # print('output 2', torch.isnan(output).int().sum())
        output = self.upsample_after_decoder_1(output, output_size = (self.args.batch_size, self.output_channels, self.output_size*2, self.output_size*2)) #[B, 288, 128, 128]
        # output = self.mish(output)
        # output = self.batch_norm_1(output)
        output = self.upsample_after_decoder_2(output, output_size = (self.args.batch_size, self.output_channels, self.args.grid_height_cells, self.args.grid_width_cells))
        # output = self.mish(output)
        # output = self.batch_norm_2(output) #[B, 288, 256, 256]
        output = self.conv(output)   #[B, 32, 256, 256]
        # print('output 3', torch.isnan(output).int().sum())
        # print('output', output.shape)
        return output.permute(0, 2, 3, 1).contiguous()