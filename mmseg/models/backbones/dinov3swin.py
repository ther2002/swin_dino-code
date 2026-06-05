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
    def __init__(self, in_channels, num_heads=8, norm_cfg=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads) ** -0.5
        self.norm = build_norm_layer(norm_cfg, in_channels)[1] if norm_cfg else nn.Identity()
        self.qkv_proj = build_conv_layer(dict(type='Conv2d'), in_channels, 3 * in_channels, 1)
        self.out_proj = build_conv_layer(dict(type='Conv2d'), in_channels, in_channels, 1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        N = H * W
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        
        qkv = self.qkv_proj(x2)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C//self.num_heads, N).transpose(2, 3)
        k = k.reshape(B, self.num_heads, C//self.num_heads, N).transpose(2, 3)
        v = v.reshape(B, self.num_heads, C//self.num_heads, N).transpose(2, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x2_attn = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        x2_attn = self.out_proj(x2_attn)
        return x1 + x2_attn


class CrossStageProgressiveFusion(nn.Module):
    def __init__(self,
                 in_channels_list,  # 需改为Swin-Base的通道列表[128,256,512,1024]
                 fusion_order='low2high',
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.num_stages = len(in_channels_list)
        self.fusion_order = fusion_order

        self.fusion_layers = torch.nn.ModuleList()
        self.channel_adapt_layers = torch.nn.ModuleList()

        for i in range(1, self.num_stages):
            if fusion_order == 'low2high':
                prev_channels = in_channels_list[i-1]
                curr_channels = in_channels_list[i]
            else:
                prev_channels = in_channels_list[i]
                curr_channels = in_channels_list[i-1]

            # 通道适配层：前序通道→当前通道（适配Swin-Base的通道递增）
            self.channel_adapt_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=prev_channels,
                    out_channels=curr_channels,
                    kernel_size=1
                )
            )

            # 融合层输入：适配后前序通道 + 当前通道（均为curr_channels）
            self.fusion_layers.append(nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=curr_channels + curr_channels,
                    out_channels=curr_channels,
                    kernel_size=3,
                    padding=1
                ),
                build_norm_layer(norm_cfg, curr_channels)[1] if norm_cfg else nn.Identity(),
                nn.ReLU() if act_cfg['type'] == 'ReLU' else nn.GELU(),
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=curr_channels,
                    out_channels=curr_channels,
                    kernel_size=1
                ),
                nn.Sigmoid()
            ))

    def forward(self, stage_feats):
        orig_sizes = [feat.shape[2:] for feat in stage_feats]
        order = list(range(self.num_stages)) if self.fusion_order == 'low2high' else list(range(self.num_stages-1, -1, -1))
        fused_results = [stage_feats[order[0]]]

        for i in range(1, self.num_stages):
            curr_stage = order[i]
            prev_fused = fused_results[-1]
            curr_feat = stage_feats[curr_stage]

            # 尺寸适配
            prev_fused_resized = F.interpolate(
                prev_fused, size=curr_feat.shape[2:], mode='bilinear', align_corners=False
            )
            # 通道适配（匹配Swin-Base当前阶段通道）
            prev_fused_adapted = self.channel_adapt_layers[i-1](prev_fused_resized)
            # 融合
            concat_feat = torch.cat([prev_fused_adapted, curr_feat], dim=1)
            attn_weight = self.fusion_layers[i-1](concat_feat)
            curr_fused = curr_feat * attn_weight + prev_fused_adapted * (1 - attn_weight)
            fused_results.append(curr_fused)

        if self.fusion_order != 'low2high':
            fused_feats = fused_results[::-1]
        else:
            fused_feats = fused_results
        return fused_feats


@MODELS.register_module()
class DINOv3SwinEncoder(BaseModule):
    def __init__(self,
                 dinov3_cfg,
                 swin_cfg,
                 feature_adapt_cfg,
                 fusion_cfg,
                 dinov3_out_channels,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.dinov3 = DINOv3Backbone(** dinov3_cfg)
        self.swin = SwinTransformer(**swin_cfg)
        self.num_stages = len(swin_cfg['depths'])  # Swin-Base为4个阶段（depths=[2,2,18,2]）
        
        self.dinov3_out_channels = self._get_dinov3_out_channels(dinov3_out_channels)
        self.swin_embed_dims = self.swin.num_features  # Swin-Base为[128,256,512,1024]
        
        # 验证通道数与阶段数匹配（Swin-Base为4阶段，dinov3_out_channels需为4个元素）
        assert len(self.dinov3_out_channels) == self.num_stages, \
            f"DINOv3阶段数({len(self.dinov3_out_channels)})与Swin不匹配({self.num_stages})"
        assert len(self.swin_embed_dims) == self.num_stages, \
            f"Swin通道列表长度({len(self.swin_embed_dims)})与阶段数({self.num_stages})不匹配"
        
        # 构建DINOv3特征适配层（适配Swin-Base的通道）
        self.feature_adapt_layers = self._build_adapt_layers(
            feature_adapt_cfg, self.dinov3_out_channels, self.swin_embed_dims)
        
        # 解析融合配置（适配Swin-Base的通道列表）
        self.fusion_type = fusion_cfg['type']
        self.use_cross_stage = self.fusion_type == 'cross_stage_progressive'
        
        if self.use_cross_stage:
            self.intra_stage_fusion_type = fusion_cfg.get('intra_stage_type', 'attention')
            self.intra_fusion_layers = self._build_intra_fusion_layers(
                fusion_cfg, self.intra_stage_fusion_type)
            # 构建跨阶段融合层（传入Swin-Base的通道列表）
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

        # 适配DINOv3（1024通道）→ Swin-Base各阶段通道（128→256→512→1024）
        for dinov3_ch, swin_ch in zip(dinov3_out_channels, swin_embed_dims):
            adapt_block = []
            # 1. 尺寸适配：DINOv3原始32×32 → 匹配Swin各阶段尺寸
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
            # 2. 通道适配：DINOv3 1024 → Swin-Base各阶段通道（如128/256等）
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
        intra_layers = nn.ModuleList()
        norm_cfg = fusion_cfg.get('norm_cfg')
        
        # 阶段内融合：输入为DINOv3适配后特征（Swin-Base通道）+ Swin特征（Swin-Base通道）
        for swin_ch in self.swin_embed_dims:
            if intra_type == 'attention':
                intra_layers.append(
                    AttentionFusion(
                        in_channels_x1=swin_ch,
                        in_channels_x2=swin_ch,
                        attn_channels=fusion_cfg.get('attn_channels', 24)
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
        fusion_layers = nn.ModuleList()
        fusion_type = fusion_cfg['type']
        norm_cfg = fusion_cfg.get('norm_cfg')
        
        for swin_ch in self.swin_embed_dims:
            if fusion_type == 'concat':
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
                fusion_layers.append(nn.Identity())
            elif fusion_type == 'attention':
                fusion_layers.append(
                    AttentionFusion(
                        in_channels_x1=swin_ch,
                        in_channels_x2=swin_ch,
                        attn_channels=fusion_cfg.get('attn_channels', 24)
                    )
                )
            elif fusion_type == 'gated':
                fusion_layers.append(
                    GatedFusion(
                        in_channels=swin_ch,
                        norm_cfg=norm_cfg
                    )
                )
            elif fusion_type == 'cross_attention':
                fusion_layers.append(
                    CrossAttentionFusion(
                        in_channels=swin_ch,
                        num_heads=fusion_cfg.get('num_heads', 8),
                        norm_cfg=norm_cfg
                    )
                )
            elif fusion_type == 'pool':
                pool_type = fusion_cfg.get('pool_type', 'max')
                fusion_layers.append(nn.MaxPool2d(kernel_size=1) if pool_type == 'max' else nn.AvgPool2d(kernel_size=1))
            else:
                raise NotImplementedError(f"不支持的融合方式: {fusion_type}")
        return fusion_layers

    def forward(self, x):
        # 获取DINOv3和Swin-Base的原始特征
        dinov3_feats = self.dinov3(x)  # 形状：[(B,1024,32,32)]*4
        swin_feats = self.swin(x)     # 形状：[(B,128,128,128), (B,256,64,64), (B,512,32,32), (B,1024,16,16)]
        # 只打印一次（避免训练刷屏）
        if not hasattr(self, '_printed_shapes'):
            print("\n[DINOv3SwinEncoder] input:", tuple(x.shape))
            print("[DINOv3] feats:")
            for i, f in enumerate(dinov3_feats):
                print(f"  D{i}: {tuple(f.shape)}")
            print("[Swin] feats:")
            for i, f in enumerate(swin_feats):
                print(f"  S{i}: {tuple(f.shape)}")
        
        assert len(dinov3_feats) == self.num_stages and len(swin_feats) == self.num_stages
        
        # 阶段内融合（DINOv3适配后 + Swin-Base）
        intra_fused_feats = []
        for i in range(self.num_stages):
            # DINOv3特征适配（尺寸→Swin尺寸，通道→Swin-Base通道）
            dinov3_adapted = self.feature_adapt_layers[i](dinov3_feats[i])
            swin_feat = swin_feats[i]
            
            # 确保尺寸一致
            if dinov3_adapted.shape[2:] != swin_feat.shape[2:]:
                dinov3_adapted = F.interpolate(
                    dinov3_adapted, size=swin_feat.shape[2:], mode='bilinear', align_corners=False
                )
            
            # 阶段内融合
            if self.use_cross_stage:
                intra_fused = self.intra_fusion_layers[i](dinov3_adapted, swin_feat)
            else:
                if self.fusion_type == 'concat':
                    intra_fused = torch.cat([dinov3_adapted, swin_feat], dim=1)
                    intra_fused = self.fusion_layers[i](intra_fused)
                elif self.fusion_type == 'add':
                    intra_fused = self.fusion_layers[i](dinov3_adapted + swin_feat)
                elif self.fusion_type in ['attention', 'gated', 'cross_attention']:
                    intra_fused = self.fusion_layers[i](dinov3_adapted, swin_feat)
                elif self.fusion_type == 'pool':
                    combined = torch.cat([dinov3_adapted, swin_feat], dim=1)
                    if isinstance(self.fusion_layers[i], nn.MaxPool2d):
                        intra_fused = F.max_pool1d(combined.flatten(2), kernel_size=2, stride=2).view_as(swin_feat)
                    else:
                        intra_fused = F.avg_pool1d(combined.flatten(2), kernel_size=2, stride=2).view_as(swin_feat)
                else:
                    raise ValueError(f"未实现的融合方式: {self.fusion_type}")
            
            intra_fused_feats.append(intra_fused)
        
        # 跨阶段融合（输出Swin-Base通道的特征）
        if self.use_cross_stage:
            cross_fused_feats = self.cross_stage_fusion(intra_fused_feats)
            return cross_fused_feats
        else:
            return intra_fused_feats