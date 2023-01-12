import torch.nn.functional as F
import torch
import torch.nn as nn
from cumm import tensorview as tv
from spconv.utils import Point2VoxelCPU3d

# NUM_FILTERS = [64]
voxel_size = [1/3.2, 1/3.2]
point_cloud_range = [-40, -20, 40, 60]  #[x_min, y_min, x_max, y_max]

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=False,
                 last_layer=False):
        super().__init__()
 
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
 
        self.part = 50000
 
    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        # torch.backends.cudnn.enabled = False
        # x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        # torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # x_max shape ：（M, 1, 64）　
        x_max = torch.max(x, dim=1, keepdim=True)[0]
 
        if self.last_vfe:
            # 返回经过简化版pointnet处理pillar的结果
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            # print('x', x.shape)
            # print('x_repeat', x_repeat.shape)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            # print('x_concat', x_concatenated.shape)
            return x_concatenated
        
class PillarVFE(nn.Module):
    """
    model_cfg:NAME: PillarVFE
                    WITH_DISTANCE: False
                    USE_ABSLOTE_XYZ: True
                    USE_NORM: True
                    NUM_FILTERS: [64]
    """
 
    def __init__(self, num_point_features, voxel_size, point_cloud_range, num_filters = 64,with_distance = True):
        super().__init__()
 
        self.use_norm = True
        # num_point_features += 4
        self.with_distance = with_distance
        if self.with_distance:
            num_point_features += 1
 
        self.num_filters = [num_filters]
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
 
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        
        self.pfn_layers = nn.ModuleList(pfn_layers)
 
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
 
    def get_output_feature_dim(self):
        return self.num_filters[-1]
 
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num:每个voxel实际点的数量(M,)
            max_num:voxel最大点的数量(32,)
        Returns:
            paddings_indicator:表明一个pillar中哪些是真实数据,哪些是填充的0数据
        """
        # 扩展一个维度，使变为（M，1）
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        # [1, 1]
        max_num_shape = [1] * len(actual_num.shape)
        # [1, -1]
        max_num_shape[axis + 1] = -1
        # (1,32)
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        # (M, 32)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator
 
    def forward(self, voxel_features, coords, num_points_per_voxel):
        # points_mean shape：(M, 1, 3)
        # print(voxel_features.shape)
        # points_mean = voxel_features[:, :, :2].sum(dim=1, keepdim=True) / num_points_per_voxel.type_as(voxel_features).view(-1, 1, 1)
        # f_cluster = voxel_features[:, :, :2] - points_mean
 
        # # 每个点云到该pillar的坐标中心点偏移量空数据 xp,yp
        # f_center = torch.zeros_like(voxel_features[:, :, :2])

        # f_center[:, :, 0] = voxel_features[:, :, 0] - (
        #         coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        # f_center[:, :, 1] = voxel_features[:, :, 1] - (
        #         coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset) 
 
        # features = [voxel_features[..., :], f_cluster, f_center]   #last dim is batch_idx

        # 如果使用距离信息
        # if self.with_distance:
        #     points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        #     features.append(points_dist)
        
        # features = torch.cat(features, dim=-1)
        
        features = voxel_features
        voxel_count = features.shape[1]
        # mask（M， 32）
        # 每个pillar中哪些是需要被保留的数据
        mask = self.get_paddings_indicator(num_points_per_voxel, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
 
        for pfn in self.pfn_layers:
            features = pfn(features)

        pillar_features = features.squeeze()
        
        return pillar_features
    
    
class PointPillarScatter(nn.Module):
    def __init__(self, args, encoding_dim):
        super().__init__()
        self.args = args
        self.num_bev_features = encoding_dim  # 64
        self.nx, self.ny, self.nz = self.args.grid_height_cells, self.args.grid_width_cells, 1  
        assert self.nz == 1
 
    def forward(self, pillar_features, coors):
        batch_spatial_features = []
        batch_size = coors[:, 0].max().int().item()
        # print('batch_size in scatter', batch_size)
            
        for batch_idx in range(batch_size+1):
            spatial_feature = torch.zeros((1, self.nx, self.ny, self.num_bev_features),dtype=pillar_features.dtype,device=pillar_features.device)  

            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, :].type(torch.long)

            pillars = pillar_features[batch_mask, :]
            # spatial_feature[0, this_coors[:, 1], this_coors[:, 2], :] = pillars
            spatial_feature[0, this_coors[:, 2], this_coors[:, 1], :] = pillars

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.cat(batch_spatial_features, 0)
 
        return batch_spatial_features