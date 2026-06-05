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
    """优化后的DINOv3 Backbone封装，支持冻结训练"""

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
        frozen_stages: int = -1,  # 新增：冻结阶段，-1表示不冻结
        norm_eval: bool = True,   # 新增：冻结时是否将norm层设为eval模式
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
        self.dinov3_model.eval()  # 初始化为eval模式，后续train()会根据需要调整

        # 原生模型参数
        self.img_size = img_size
        self.patch_size = self.dinov3_model.patch_size
        self.n_blocks = self.dinov3_model.n_blocks
        self.n_storage_tokens = self.dinov3_model.n_storage_tokens  # 寄存器token数量
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

        # 执行冻结逻辑
        self._freeze_stages()

        self._is_init = True

    def _freeze_stages(self):
        """冻结指定阶段的参数，参考ResNet/Swin等backbone的实现"""
        if self.frozen_stages >= 0:
            # 冻结整个backbone
            self.dinov3_model.eval()  # 冻结时保持eval模式
            for param in self.dinov3_model.parameters():
                param.requires_grad = False

            # 如果需要部分冻结（可选扩展）
            # 例如：只冻结前n个block
        # if self.frozen_stages < self.n_blocks:
        #     for i in range(self.frozen_stages):
        #         block = self.dinov3_model.blocks[i]
        #         block.eval()
        #         for param in block.parameters():
        #             param.requires_grad = False

    def train(self, mode=True):
        """重写train方法，确保冻结的层在训练时保持eval模式"""
        super().train(mode)
        if mode and self.norm_eval:
            # 训练模式下，冻结的norm层保持eval
            for m in self.dinov3_model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()
        # 处理冻结阶段
        if self.frozen_stages >= 0 and mode:
            self.dinov3_model.eval()

    def forward(self, x):
        """前向传播，确保梯度能正常传递到下游层（即使backbone被冻结）"""
        x = self.patch_adapter(x)
        input_h, input_w = x.shape[2], x.shape[3]  # 适配后的输入尺寸

        # 不使用inference_mode，确保梯度能流过backbone（即使参数被冻结）
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
                patch_feat, class_token = feat  # 解包特征图和class token
                cls_tokens.append(class_token)
            else:
                patch_feat = feat  # 仅特征图

            # 验证特征图尺寸与输入尺寸的匹配性
            b, c, h, w = patch_feat.shape
            assert abs(h * self.patch_size - input_h) <= self.patch_size, \
                f"特征图高度{h}与输入高度{input_h}不匹配（patch_size={self.patch_size}）"
            assert abs(w * self.patch_size - input_w) <= self.patch_size, \
                f"特征图宽度{w}与输入宽度{input_w}不匹配（patch_size={self.patch_size}）"

            features.append(patch_feat.contiguous())

        if self.output_cls_token:
            return features, cls_tokens
        return features