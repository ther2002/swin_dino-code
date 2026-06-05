import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from .swin import SwinTransformer
from .dinov3 import DINOv3Backbone  # 假设存在该类
from mmcv.cnn import build_activation_layer, build_norm_layer, build_conv_layer


class AttentionFusion(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, attn_channels=24):
        super().__init__()
        # 接受注意力通道数参数，默认24
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels_x1 + in_channels_x2, attn_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(attn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        attn_weight = self.attn(torch.cat([x1, x2], dim=1))
        return attn_weight * x1 + (1 - attn_weight) * x2


class GatedFusion(nn.Module):
    """门控融合模块：类似LSTM门控机制控制特征比例"""
    def __init__(self, in_channels, norm_cfg=None):
        super().__init__()
        self.gate = nn.Sequential(
            build_conv_layer(dict(type='Conv2d'), 2 * in_channels, in_channels, 1),
            build_norm_layer(norm_cfg, in_channels)[1] if norm_cfg else nn.Identity(),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gate = self.gate(torch.cat([x1, x2], dim=1))
        return gate * x1 + (1 - gate) * x2


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合：通过Transformer交叉注意力交互特征"""
    def __init__(self, in_channels, num_heads=8, norm_cfg=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) **-0.5
        self.norm = build_norm_layer(norm_cfg, in_channels)[1] if norm_cfg else nn.Identity()
        
        # 交叉注意力线性层
        self.qkv_proj = build_conv_layer(dict(type='Conv2d'), in_channels, 3 * in_channels, 1)
        self.out_proj = build_conv_layer(dict(type='Conv2d'), in_channels, in_channels, 1)

    def forward(self, x1, x2):
        # x1: DINOv3特征, x2: Swin特征
        B, C, H, W = x1.shape
        N = H * W
        
        # 统一归一化
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        
        # 生成查询(来自x1)、键值(来自x2)
        qkv = self.qkv_proj(x2)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 每个都是[B, C, H, W]
        
        # 转换为注意力格式 [B, num_heads, N, C//num_heads]
        q = q.reshape(B, self.num_heads, C//self.num_heads, N).transpose(2, 3)  # [B, H, N, C/H]
        k = k.reshape(B, self.num_heads, C//self.num_heads, N).transpose(2, 3)
        v = v.reshape(B, self.num_heads, C//self.num_heads, N).transpose(2, 3)
        
        # 计算注意力权重 (x1作为查询关注x2的特征)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        
        # 应用注意力到x2特征
        x2_attn = (attn @ v).transpose(2, 3).reshape(B, C, H, W)  # [B, C, H, W]
        x2_attn = self.out_proj(x2_attn)
        
        # 融合结果
        return x1 + x2_attn


class CrossStageProgressiveFusion(nn.Module):
    """跨阶段渐进融合模块：逐步融合不同阶段的特征
    
    支持从低阶到高阶或高阶到低阶的渐进式融合，每个阶段特征与前序融合结果
    通过注意力机制进行交互，保留各阶段特征的细节与语义信息
    """
    def __init__(self, 
                 in_channels_list,  # 各阶段输入通道数列表
                 fusion_order='low2high',  # 融合顺序：低到高或高到低
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.fusion_order = fusion_order
        self.num_stages = len(in_channels_list)
        
        # 通道统一卷积：将各阶段特征调整到相同通道数
        self.channel_unify = nn.ModuleList()
        self.out_channels = in_channels_list[-1] if fusion_order == 'low2high' else in_channels_list[0]
        for ch in in_channels_list:
            self.channel_unify.append(
                nn.Sequential(
                    build_conv_layer(dict(type='Conv2d'), ch, self.out_channels, 1),
                    build_norm_layer(norm_cfg, self.out_channels)[1] if norm_cfg else nn.Identity(),
                    build_activation_layer(act_cfg) if act_cfg else nn.Identity()
                )
            )
        
        # 阶段间融合注意力：学习当前阶段与前序融合结果的权重
        self.stage_attn = nn.ModuleList()
        for _ in range(self.num_stages - 1):
            self.stage_attn.append(
                nn.Sequential(
                    build_conv_layer(dict(type='Conv2d'), 2 * self.out_channels, self.out_channels, 3, padding=1),
                    build_norm_layer(norm_cfg, self.out_channels)[1] if norm_cfg else nn.Identity(),
                    nn.ReLU(),
                    build_conv_layer(dict(type='Conv2d'), self.out_channels, self.out_channels, 1),
                    nn.Sigmoid()
                )
            )

    def forward(self, stage_feats):
        """
        Args:
            stage_feats (list[Tensor]): 各阶段融合后的特征列表，形状为
                [(B, C1, H1, W1), (B, C2, H2, W2), ..., (B, Cn, Hn, Wn)]
                其中H1 > H2 > ... > Hn（分辨率从高到低）
        
        Returns:
            list[Tensor]: 跨阶段融合后的各阶段特征列表
        """
        # 1. 统一各阶段特征通道数并记录原始尺寸
        unified_feats = []
        original_sizes = []
        for i, feat in enumerate(stage_feats):
            original_sizes.append(feat.shape[2:])  # 保存原始空间尺寸
            unified = self.channel_unify[i](feat)
            unified_feats.append(unified)
        
        # 2. 确定融合顺序和目标尺寸
        if self.fusion_order == 'low2high':
            # 从低阶(小尺寸)到高阶(大尺寸)融合，以最高阶尺寸为目标
            order = list(range(self.num_stages))
            target_sizes = original_sizes
        else:
            # 从高阶(大尺寸)到低阶(小尺寸)融合，以最低阶尺寸为目标
            order = list(range(self.num_stages-1, -1, -1))
            target_sizes = original_sizes[::-1]
        
        # 3. 渐进式融合
        fused_results = []
        current_fused = unified_feats[order[0]]  # 初始化为第一个阶段特征
        fused_results.append(current_fused)
        
        for i in range(1, self.num_stages):
            # 获取当前阶段特征并调整尺寸
            current_stage = order[i]
            current_feat = unified_feats[current_stage]
            current_feat = F.interpolate(
                current_feat,
                size=current_fused.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 计算注意力权重
            attn_weight = self.stage_attn[i-1](torch.cat([current_fused, current_feat], dim=1))
            
            # 注意力加权融合
            current_fused = attn_weight * current_feat + (1 - attn_weight) * current_fused
            fused_results.append(current_fused)
        
        # 4. 恢复各阶段原始尺寸并整理顺序
        if self.fusion_order == 'low2high':
            # 按原始顺序输出，恢复各自尺寸
            output_feats = []
            for i in range(self.num_stages):
                output = F.interpolate(
                    fused_results[i],
                    size=original_sizes[i],
                    mode='bilinear',
                    align_corners=False
                )
                output_feats.append(output)
        else:
            # 逆序恢复
            output_feats = []
            for i in range(self.num_stages):
                output = F.interpolate(
                    fused_results[self.num_stages-1 - i],
                    size=original_sizes[i],
                    mode='bilinear',
                    align_corners=False
                )
                output_feats.append(output)
        
        return output_feats


@MODELS.register_module()
class DINOv3SwinEncoder(BaseModule):
    """DINOv3与Swin Transformer的融合编码器（支持多种融合方式）"""
    def __init__(self,
                 dinov3_cfg,
                 swin_cfg,
                 feature_adapt_cfg,
                 fusion_cfg,
                 dinov3_out_channels,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 初始化backbone
        self.dinov3 = DINOv3Backbone(** dinov3_cfg)
        self.swin = SwinTransformer(**swin_cfg)
        self.num_stages = len(swin_cfg['depths'])
        
        # 处理输出通道配置
        self.dinov3_out_channels = self._get_dinov3_out_channels(dinov3_out_channels)
        self.swin_embed_dims = self.swin.num_features
        
        # 验证配置合法性
        assert len(self.dinov3_out_channels) == self.num_stages, \
            f"DINOv3阶段数({len(self.dinov3_out_channels)})与Swin不匹配({self.num_stages})"
        assert len(self.swin_embed_dims) == self.num_stages, \
            f"Swin通道列表长度({len(self.swin_embed_dims)})与阶段数({self.num_stages})不匹配"
        
        # 构建特征适配层和融合层
        self.feature_adapt_layers = self._build_adapt_layers(
            feature_adapt_cfg, self.dinov3_out_channels, self.swin_embed_dims)
        
        # 解析融合配置
        self.fusion_type = fusion_cfg['type']
        self.use_cross_stage = self.fusion_type == 'cross_stage_progressive'
        
        if self.use_cross_stage:
            # 跨阶段融合需要先进行单阶段融合
            self.intra_stage_fusion_type = fusion_cfg.get('intra_stage_type', 'attention')
            self.intra_fusion_layers = self._build_intra_fusion_layers(
                fusion_cfg, self.intra_stage_fusion_type)
            # 构建跨阶段渐进融合层
            self.cross_stage_fusion = CrossStageProgressiveFusion(
                in_channels_list=self.swin_embed_dims,
                fusion_order=fusion_cfg.get('fusion_order', 'low2high'),
                norm_cfg=fusion_cfg.get('norm_cfg'),
                act_cfg=fusion_cfg.get('act_cfg', dict(type='ReLU'))
            )
        else:
            self.fusion_layers = self._build_fusion_layers(fusion_cfg)

    def _get_dinov3_out_channels(self, dinov3_out_channels):
        if isinstance(dinov3_out_channels, list):
            return dinov3_out_channels
        elif isinstance(dinov3_out_channels, int):
            return [dinov3_out_channels] * self.num_stages
        else:
            raise ValueError("dinov3_out_channels必须是列表或整数")

    def _build_adapt_layers(self, feature_adapt_cfg, dinov3_out_channels, swin_embed_dims):
        adapt_layers = nn.ModuleList()
        conv_type = feature_adapt_cfg.get('conv_type', 'Conv2d')
        norm_cfg = feature_adapt_cfg.get('norm_cfg')
        act_cfg = feature_adapt_cfg.get('act_cfg')

        for dinov3_ch, swin_ch in zip(dinov3_out_channels, swin_embed_dims):
            adapt_block = []
            # 1. 动态调整空间尺寸
            adapt_block.append(
                build_conv_layer(
                    dict(type=conv_type),
                    in_channels=dinov3_ch,
                    out_channels=dinov3_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            # 2. 调整通道数
            adapt_block.append(
                build_conv_layer(
                    dict(type=conv_type),
                    in_channels=dinov3_ch,
                    out_channels=swin_ch,
                    kernel_size=1
                )
            )
            # 3. 归一化和激活
            if norm_cfg is not None:
                norm_name, norm_layer = build_norm_layer(norm_cfg, swin_ch)
                adapt_block.append(norm_layer)
            if act_cfg is not None:
                adapt_block.append(build_activation_layer(act_cfg))

            adapt_layers.append(nn.Sequential(*adapt_block))
        return adapt_layers

    def _build_intra_fusion_layers(self, fusion_cfg, intra_type):
        """构建阶段内融合层（跨阶段融合的前置步骤）"""
        intra_layers = nn.ModuleList()
        norm_cfg = fusion_cfg.get('norm_cfg')
        
        for swin_ch in self.swin_embed_dims:
            if intra_type == 'attention':
                intra_layers.append(
                    AttentionFusion(
                        in_channels_x1=swin_ch,  # 修正参数名称
                        in_channels_x2=swin_ch,  # 新增第二个输入通道参数
                        attn_channels=fusion_cfg.get('attn_channels', 24)  # 注意力通道数，默认24
                    )
                )
            elif intra_type == 'gated':
                intra_layers.append(
                    GatedFusion(
                        in_channels=swin_ch,
                        norm_cfg=norm_cfg
                    )
                )
            elif intra_type == 'add':
                intra_layers.append(nn.Identity())
            elif intra_type == 'concat':
                intra_layers.append(
                    nn.Sequential(
                        build_conv_layer(
                            fusion_cfg.get('conv_cfg', dict(type='Conv2d')),
                            2 * swin_ch,
                            swin_ch,
                            kernel_size=1
                        ),
                        build_norm_layer(norm_cfg, swin_ch)[1] if norm_cfg else nn.Identity(),
                        build_activation_layer(fusion_cfg.get('act_cfg', dict(type='ReLU')))
                    )
                )
            else:
                raise NotImplementedError(f"不支持的阶段内融合方式: {intra_type}")
        return intra_layers

    def _build_fusion_layers(self, fusion_cfg):
        """构建多种融合方式的层"""
        fusion_layers = nn.ModuleList()
        fusion_type = fusion_cfg['type']
        norm_cfg = fusion_cfg.get('norm_cfg')
        
        for swin_ch in self.swin_embed_dims:
            if fusion_type == 'concat':
                # 拼接后压缩通道
                fusion_layers.append(
                    nn.Sequential(
                        build_conv_layer(
                            fusion_cfg.get('conv_cfg', dict(type='Conv2d')),
                            2 * swin_ch,
                            swin_ch,
                            kernel_size=1
                        ),
                        build_norm_layer(norm_cfg, swin_ch)[1] if norm_cfg else nn.Identity(),
                        build_activation_layer(fusion_cfg.get('act_cfg', dict(type='ReLU')))
                    )
                )
            elif fusion_type == 'add':
                # 直接相加（需确保通道和尺寸一致）
                fusion_layers.append(nn.Identity())
            elif fusion_type == 'attention':
                # 注意力加权融合
                fusion_layers.append(
                    AttentionFusion(
                        in_channels_x1=swin_ch,  # 修正参数名称
                        in_channels_x2=swin_ch,  # 新增第二个输入通道参数
                        attn_channels=fusion_cfg.get('attn_channels', 24)  # 注意力通道数，默认24
                    )
                )
            elif fusion_type == 'gated':
                # 门控机制融合
                fusion_layers.append(
                    GatedFusion(
                        in_channels=swin_ch,
                        norm_cfg=norm_cfg
                    )
                )
            elif fusion_type == 'cross_attention':
                # 交叉注意力融合
                fusion_layers.append(
                    CrossAttentionFusion(
                        in_channels=swin_ch,
                        num_heads=fusion_cfg.get('num_heads', 8),
                        norm_cfg=norm_cfg
                    )
                )
            elif fusion_type == 'pool':
                # 池化融合（取最大/平均）
                pool_type = fusion_cfg.get('pool_type', 'max')
                if pool_type == 'max':
                    fusion_layers.append(nn.MaxPool2d(kernel_size=1))
                else:
                    fusion_layers.append(nn.AvgPool2d(kernel_size=1))
            else:
                raise NotImplementedError(f"不支持的融合方式: {fusion_type}")
        return fusion_layers

    def forward(self, x):
        # 打印输入图像形状
        # print(f"输入图像形状: {x.shape}")
        
        # 获取原始特征
        dinov3_feats = self.dinov3(x)
        swin_feats = self.swin(x)
        assert len(dinov3_feats) == self.num_stages and len(swin_feats) == self.num_stages
        
        # 打印backbone输出的原始特征形状
        # print("\n=== 原始特征形状 ===")
        # for i in range(self.num_stages):
        #     print(f"阶段{i+1} - DINOv3原始特征: {dinov3_feats[i].shape}")
        #     print(f"阶段{i+1} - Swin原始特征: {swin_feats[i].shape}")
        
        # 阶段内特征融合（单阶段内的DINOv3与Swin特征融合）
        intra_fused_feats = []
        # print("\n=== 阶段内融合特征形状 ===")
        for i in range(self.num_stages):
            # print(f"\n----- 阶段{i+1} -----")
            # 适配DINOv3特征
            dinov3_adapted = self.feature_adapt_layers[i](dinov3_feats[i])
            # print(f"DINOv3适配后特征: {dinov3_adapted.shape}")
            
            # Swin原始特征
            swin_feat = swin_feats[i]
            # print(f"Swin特征: {swin_feat.shape}")
            
            # 确保空间尺寸一致
            if dinov3_adapted.shape[2:] != swin_feat.shape[2:]:
                dinov3_adapted = F.interpolate(
                    dinov3_adapted,
                    size=swin_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                # print(f"DINOv3适配后(插值调整)特征: {dinov3_adapted.shape}")
            
            # 阶段内融合
            if self.use_cross_stage:
                intra_fused = self.intra_fusion_layers[i](dinov3_adapted, swin_feat)
            else:
                if self.fusion_type == 'concat':
                    intra_fused = torch.cat([dinov3_adapted, swin_feat], dim=1)
                    # print(f"拼接后特征: {intra_fused.shape}")
                    intra_fused = self.fusion_layers[i](intra_fused)
                elif self.fusion_type == 'add':
                    intra_fused = self.fusion_layers[i](dinov3_adapted + swin_feat)
                elif self.fusion_type in ['attention', 'gated', 'cross_attention']:
                    intra_fused = self.fusion_layers[i](dinov3_adapted, swin_feat)
                elif self.fusion_type == 'pool':
                    combined = torch.cat([dinov3_adapted, swin_feat], dim=1)
                    # print(f"拼接后特征(池化前): {combined.shape}")
                    if isinstance(self.fusion_layers[i], nn.MaxPool2d):
                        intra_fused = F.max_pool1d(
                            combined.flatten(2), kernel_size=2, stride=2
                        ).view_as(swin_feat)
                    else:
                        intra_fused = F.avg_pool1d(
                            combined.flatten(2), kernel_size=2, stride=2
                        ).view_as(swin_feat)
                else:
                    raise ValueError(f"未实现的融合方式: {self.fusion_type}")
            
            # print(f"阶段内融合后特征: {intra_fused.shape}")
            intra_fused_feats.append(intra_fused)
        
        # 跨阶段渐进融合（如果启用）
        if self.use_cross_stage:
            # print("\n=== 跨阶段渐进融合 ===")
            cross_fused_feats = self.cross_stage_fusion(intra_fused_feats)
            # for i in range(self.num_stages):
                # print(f"跨阶段融合后阶段{i+1}特征: {cross_fused_feats[i].shape}")
            return cross_fused_feats
        else:
            return intra_fused_feats