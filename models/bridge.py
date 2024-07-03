# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from functools import partial

from timm.models.layers import drop_path, to_2tuple
import numpy as np

#将模型的参数全0处理
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
#随机深度技术,正则化,前向传播时忽略这些路径
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

#两层全连接,一次激活函数一次dropout
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

#Attention模块
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, att_mask=None, return_attn_map=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        logit = (q @ k.transpose(-2, -1))

        if att_mask is not None:
            if att_mask.dim() == 3:
                att_mask = att_mask.unsqueeze(1)
            elif att_mask.dim() == 2:
                att_mask = att_mask.unsqueeze(1).unsqueeze(1)
            logit = logit.masked_fill(att_mask == 0, float('-inf'))

        real_attn = logit.softmax(dim=-1)
        attn = self.attn_drop(real_attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn_map:
            return x, logit
        else:
            return x
#将图像转换为嵌入的嵌入的图像序列块
#输入:3*224*224
#以16*16为一个块,有196个嵌入块,每个块的维度为768
#输出:1*196*768,适合Transformer模型操作
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = to_2tuple(img_size)
        if not isinstance(patch_size, tuple):
            patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.fc = nn.Linear(64,32)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.fc(self.proj(x).flatten(2)).transpose(1, 2)
        return x

# input_image = torch.randn(1, 3, 224, 224)  # 形状为 (B, C, H, W)
# model = PatchEmbed()
# summary(model=model, input_size=(1,3,224,224), device="cpu")
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, att_mask=None, return_attn_map=False):
        if self.gamma_1 is None:
            if return_attn_map:
                attn_x, attn_map = self.attn(self.norm1(x), att_mask, return_attn_map)
            else:
                attn_x = self.attn(self.norm1(x), att_mask)
            x = x + self.drop_path(attn_x)
            # x = x + self.drop_path(self.attn(self.norm1(x), att_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if return_attn_map:
                attn_x, attn_map = self.attn(self.norm1(x), att_mask, return_attn_map)
            else:
                attn_x = self.attn(self.norm1(x), att_mask)
            x = x + self.drop_path(self.gamma_1 * attn_x)
            # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), att_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if return_attn_map:
            return x, attn_map
        else:
            return x
# 创建一个随机的输入张量
# input_tensor = torch.randn(1, 16, 768)  # 形状为 (B, N, D)，其中 B 是批量大小，N 是序列长度，D 是特征维度
# # 实例化 Block
# block = Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1, init_values=0.1)
# summary(model=block, input_size=(1,16,768), device="cpu")

# 位置编码
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

#Bridge结构
class Bridge(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
                 use_learnable_pos_emb=False, use_mean_pooling=False, init_scale=0.001, return_feat_map=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 32, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(32, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.zero_convs = zero_module(nn.Linear(512, 512))
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)


    def forward(self, x, mask=None, enc_feat=None):
        # 嵌入
        x = self.patch_embed(x)
        ## 1,196,768
        B, N, C = x.shape
        ##添加位置编码
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        #12块
        for blk in self.blocks:
            x = blk(x)
        #最后初始化参数为全0的全连接层
        x = self.zero_convs(x)
        return x
# 实例化 Bridge 模型
# bridge = Bridge(img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12)
# data = torch.randn(1,3,224,224)
# # 使用 summary 函数查看模型摘要
# bridge(data)