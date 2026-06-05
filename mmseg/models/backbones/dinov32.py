import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmseg.registry import MODELS


def _get_backbone_out_indices(n_blocks):
    """获取DINOv3原生的特征输出阶段索引（FOUR_EVEN_INTERVALS模式）"""
    if n_blocks == 12:  # ViT-S/16, ViT-S+/16, ViT-B/16
        return [2, 5, 8, 11]
    elif n_blocks == 24:  # ViT-L/16
        return [4, 11, 17, 23]  # 兼容原生代码的历史实现
    elif n_blocks == 40:  # ViT-7B/16
        return [9, 19, 29, 39]
    else:
        raise ValueError(f"未支持的DINOv3模型总块数: {n_blocks}")


class CenterPadding(nn.Module):
    """原生实现的中心 padding，确保输入尺寸为patch_size的倍数"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return torch.nn.functional.pad(
            x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant'
        )


class InterpolateAdaptation(nn.Module):
    """通过插值将输入尺寸调整为patch_size的倍数"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        h, w = x.shape[-2:]
        target_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        target_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        return torch.nn.functional.interpolate(
            x, size=(target_h, target_w), mode='bilinear', align_corners=False
        )


@MODELS.register_module()
class DINOv3Backbone(BaseModule):
    """优化后的DINOv3 Backbone封装，支持指定层冻结与微调"""

    def __init__(
        self,
        model_name,
        repo_dir,
        weights,
        img_size=896,
        in_channels=3,
        init_cfg=None,
        adapt_patch_size: str = "center_padding",
        output_cls_token: bool = False,
        frozen_stages: int = -1,  # 冻结前N层：-1=不冻结，0=冻结patch_embed，1+=冻结前N个block
        norm_eval: bool = True,   # 冻结时是否将norm层设为eval模式
        **kwargs,
    ):
        super().__init__(init_cfg)
        if in_channels != 3:
            raise NotImplementedError("DINOv3仅支持3通道输入图像")

        # 加载DINOv3模型
        self.dinov3_model = torch.hub.load(
            repo_dir,
            model_name,
            source='local',
            weights=weights,
            check_hash=False,** kwargs
        )

        # 原生模型参数
        self.img_size = img_size
        self.patch_size = self.dinov3_model.patch_size
        self.n_blocks = self.dinov3_model.n_blocks  # 总block数量（如12/24/40）
        self.n_storage_tokens = self.dinov3_model.n_storage_tokens
        self.embed_dim = self.dinov3_model.embed_dim
        self.out_indices = _get_backbone_out_indices(self.n_blocks)
        self.output_cls_token = output_cls_token
        
        # 冻结相关参数
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 输入尺寸适配策略
        if adapt_patch_size == "center_padding":
            self.patch_adapter = CenterPadding(self.patch_size)
        elif adapt_patch_size == "interpolate":
            self.patch_adapter = InterpolateAdaptation(self.patch_size)
        elif adapt_patch_size == "none":
            self.patch_adapter = nn.Identity()
        else:
            raise ValueError(f"不支持的输入适配策略: {adapt_patch_size}")

        # 验证冻结参数合法性
        if self.frozen_stages < -1:
            raise ValueError(f"frozen_stages必须≥-1，当前为{self.frozen_stages}")
        if self.frozen_stages > self.n_blocks:
            raise ValueError(f"frozen_stages({self.frozen_stages})不能超过总block数({self.n_blocks})")

        # 执行冻结逻辑
        self._freeze_stages()

        self._is_init = True

    def _freeze_stages(self):
        """精细化冻结逻辑：支持冻结patch_embed和前N个block"""
        # 1. 冻结patch_embed（图像→patch的投影层）
        if self.frozen_stages >= 0:
            self.dinov3_model.patch_embed.eval()
            for param in self.dinov3_model.patch_embed.parameters():
                param.requires_grad = False

        # 2. 冻结前N个block（若frozen_stages≥1）
        if self.frozen_stages >= 1:
            for i in range(self.frozen_stages):
                block = self.dinov3_model.blocks[i]
                block.eval()  # 冻结时设为eval模式（固定norm统计量）
                for param in block.parameters():
                    param.requires_grad = False  # 禁用梯度

        # 3. 未冻结的层保持train模式（允许微调）
        if self.frozen_stages < self.n_blocks:
            for i in range(max(self.frozen_stages, 0), self.n_blocks):
                block = self.dinov3_model.blocks[i]
                block.train()  # 确保未冻结层处于train模式
                for param in block.parameters():
                    param.requires_grad = True  # 启用梯度

    def train(self, mode=True):
        """重写train方法，确保冻结层在训练时保持eval模式"""
        super().train(mode)
        if mode and self.norm_eval:
            # 训练模式下，冻结层的norm层保持eval（避免更新统计量）
            # 冻结的patch_embed
            if self.frozen_stages >= 0:
                for m in self.dinov3_model.patch_embed.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                        m.eval()
            # 冻结的block
            if self.frozen_stages >= 1:
                for i in range(self.frozen_stages):
                    for m in self.dinov3_model.blocks[i].modules():
                        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                            m.eval()

    def forward(self, x):
        """前向传播，确保梯度正常传递给未冻结层"""
        x = self.patch_adapter(x)
        input_h, input_w = x.shape[2], x.shape[3]

        # 不使用inference_mode，确保未冻结层的梯度可跟踪
        intermediate_features = self.dinov3_model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=self.output_cls_token,
            norm=True
        )

        features = []
        cls_tokens = [] if self.output_cls_token else None

        for feat in intermediate_features:
            if self.output_cls_token:
                patch_feat, class_token = feat
                cls_tokens.append(class_token)
            else:
                patch_feat = feat

            # 验证特征尺寸
            b, c, h, w = patch_feat.shape
            assert abs(h * self.patch_size - input_h) <= self.patch_size, \
                f"特征图高度{h}与输入高度{input_h}不匹配（patch_size={self.patch_size}）"
            assert abs(w * self.patch_size - input_w) <= self.patch_size, \
                f"特征图宽度{w}与输入宽度{input_w}不匹配（patch_size={self.patch_size}）"

            features.append(patch_feat.contiguous())

        if self.output_cls_token:
            return features, cls_tokens
        return features