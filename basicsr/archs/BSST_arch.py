# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.archs.propainter.sparse_transformer import TemporalSparseTransformerBlock,SoftComp,SoftSplit
from basicsr.archs.propainter.recurrent_flow_completion import RecurrentFlowCompleteNet
from basicsr.archs.gshift_arch import Encoder_shift_block,CAB,PixelShufflePack,SkipUpSample,conv,CAB1,CAB2


class SecondOrderDeformableAlignmentWithBlurmapguide(ModulatedDeformConvPack):
    """Second-order deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignmentWithBlurmapguide, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4 + 5, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2, blur_map_1,blur_map_2):
        # print("input_mean:{},std:{}".format(x.mean(),x.std()))
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)


        mask_1,mask_2 = torch.chunk(mask, 2, dim=1)
        blur_map_1 = (2*blur_map_1 - 1)*0.1
        blur_map_2 = (2*blur_map_2 - 1)*0.1
        mask_1 = mask_1 + blur_map_1.repeat(1, mask_1.size(1), 1, 1)
        mask_2 = mask_2 + blur_map_2.repeat(1, mask_1.size(1), 1, 1)

        mask = torch.cat([mask_1,mask_2], dim=1)
        
        # mask
        mask = torch.sigmoid(mask)
        # mask = None
        output = torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                            self.dilation, mask)
        # print("output_mean:{},std:{}".format(output.mean(),output.std()))
        return output

def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)
def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).type_as(flow_fw)
    return fb_valid_fw,flow_diff_fw

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """ Window based multi-head self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, -1, dtype=q.dtype)  # Don't use attn.dtype after addition!
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index


class STL(nn.Module):
    """ Swin Transformer Layer (STL).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class STG(nn.Module):
    """ Swin Transformer Group (STG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[2, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            STL(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB).

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, **kwargs):
        super(RSTB, self).__init__()
        self.input_resolution = kwargs['input_resolution']

        self.residual_group = STG(**kwargs)
        self.linear = nn.Linear(kwargs['dim'], kwargs['dim'])

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class RSTBWithInputConv(nn.Module):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1, groups=1, num_blocks=2, **kwargs):
        super().__init__()

        main = []
        main += [Rearrange('n d c h w -> n c d h w'),
                 nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups),
                 Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n c d h w')]

        # RSTB blocks
        kwargs['use_checkpoint_attn'] = kwargs.pop('use_checkpoint_attn')[0]
        kwargs['use_checkpoint_ffn'] = kwargs.pop('use_checkpoint_ffn')[0]
        main.append(make_layer(RSTB, num_blocks, **kwargs))

        main += [Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n d c h w')]

        self.main = nn.Sequential(*main)

    def forward(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """
        return self.main(x)

class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.PixelShuffle(2))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.PixelShuffle(3))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def get_blur_map(flow_forwards,flow_backwards):
    # flows: b,t-1,2,h,w
    b,t,c,h,w = flow_forwards.shape
    blur_map_first = length_sq(flow_backwards[:,0,...]).reshape(b,1,1,h,w)
    blur_map_last = length_sq(flow_forwards[:,-1,...]).reshape(b,1,1,h,w)
    blur_map = (length_sq(flow_backwards[:,1:,...].reshape(b*(t-1),2,h,w)) + length_sq(flow_forwards[:,:-1,...].reshape(b*(t-1),2,h,w)))/2
    blur_map = blur_map.reshape(b,t-1,1,h,w)
    blur_map = torch.cat([blur_map_first,blur_map,blur_map_last],dim=1)
    blur_max,_ = torch.max(blur_map.reshape(b,-1),dim=-1)
    blur_min,_ = torch.min(blur_map.reshape(b,-1),dim=-1)
    blur_st = (blur_map - blur_min.reshape(b,1,1,1,1))/(blur_max-blur_min).reshape(b,1,1,1,1)
    return blur_st
@ARCH_REGISTRY.register()
class BSST(nn.Module):
    """
        Args:
            upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 1.
            img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
            window_size (int | tuple(int)): Window size. Default: (2,8,8).
            depths (list[int]): Depths of each RSTB.
            embed_dims (list[int]): Number of linear projection output channels.
            num_heads (list[int]): Number of attention head of each stage.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
            inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
            max_residue_magnitude (int): The maximum magnitude of the offset
            residue. Default: 5. 
            cpu_cache_length: When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100. 
    """
    def __init__(self,
                 upscale=1,
                 clip_size=2,
                 img_size=[2, 64, 64],
                 window_size=[2, 8, 8],
                 depths=[2, 2, 2],
                 embed_dims=[192, 192, 192],
                 num_heads=[6, 6, 6],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 inputconv_groups=[1, 3, 3, 3, 3, 3],
                 max_residue_magnitude=5,
                 cpu_cache_length=150
                 ):
        super().__init__()
        self.upscale = upscale
        self.cpu_cache_length = cpu_cache_length
        self.mid_channels = embed_dims[0]
        
        self.feat_extract = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
        CAB(64, 3, 4, bias=False, act=nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        n_feat = 64
        kernel_size = 3
        reduction = 4
        bias = False
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.concat = nn.Sequential(nn.Conv2d(3, n_feat, 3, 1, 1, bias=False), act,CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.down01 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, 2, 1, bias=False), act)

        self.encoder_level1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level1_1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level2 = CAB(192, kernel_size, reduction, bias=bias, act=act)
        self.encoder_level2_1 = CAB(192, kernel_size, reduction, bias=bias, act=act)
        
        
        self.down12 = nn.Sequential(nn.Conv2d(n_feat, 192, 3, 2, 1, bias=False), act)
        

        self.decoder_level1 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_1 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level1_2 = Encoder_shift_block(n_feat, kernel_size, reduction, bias)
        self.decoder_level2 = Encoder_shift_block(192, kernel_size, reduction, bias)
        self.decoder_level2_1 = Encoder_shift_block(192, kernel_size, reduction, bias)
        self.decoder_level2_2 = Encoder_shift_block(192, kernel_size, reduction, bias)
        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.upsample0 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
        self.skip_conv = CAB(64, kernel_size, reduction, bias=bias, act=act) #conv(n_feat, n_feat, kernel_size, bias=bias)
        self.out_conv = CAB(64, kernel_size, reduction, bias=bias, act=act)
        self.last_conv = conv(64, 3, kernel_size, bias=bias)
        self.conv_hr0 = conv(64, 64, kernel_size, bias=bias)

        self.up21 = SkipUpSample(n_feat, 128)
       



        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False
        self.is_with_alignment = True
        
        # recurrent feature refinement
        self.backbone = nn.ModuleDict()
        self.backbone_clip = nn.ModuleDict()
        self.deform_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1','backward_2', 'forward_2']
        for i, module in enumerate(modules):
            # deformable attention
            self.deform_align[module] = SecondOrderDeformableAlignmentWithBlurmapguide(
                    2 * embed_dims[0],
                    embed_dims[0],
                    3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * embed_dims[0],embed_dims[0],3,1,1),
                self.act,
                CAB1(embed_dims[0],3,4,False,self.act)
            )
        
        
        
        self.reconstruction = RSTBWithInputConv(
                                               in_channels= 5*embed_dims[0],
                                               kernel_size=(1, 3, 3),
                                               groups=inputconv_groups[5],
                                               num_blocks=2,
                                               dim=embed_dims[2],
                                               input_resolution=[2, img_size[1], img_size[2]],
                                               depth=depths[2],
                                               num_heads=num_heads[2],
                                               window_size=[2, window_size[1], window_size[2]],
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               norm_layer=norm_layer,
                                               use_checkpoint_attn=[False],
                                               use_checkpoint_ffn=[False]
                                               )

        self.conv_before_upsampler = nn.Sequential(
                                                  nn.Conv3d(embed_dims[-1], 64, kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                                  )
        self.conv_before_input = nn.Sequential(   Rearrange('n d c h w -> n c d h w'),
                                                  nn.Conv3d(embed_dims[-1], embed_dims[-1], kernel_size=(1, 3, 3),
                                                            padding=(0, 1, 1)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                                  nn.Conv3d(embed_dims[-1], embed_dims[-1], kernel_size=(1, 3, 3),
                                                            padding=(0, 1, 1)),
                                                  Rearrange('n c d h w -> n d c h w')
                                                  )
        self.upsampler = Upsample(4, 64)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        
        self.init_SparseTransformer()
        
        self.blur_motion_refine = RecurrentFlowCompleteNet()
        
    def init_weights(self, init_type='normal', gain=0.01):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1 or classname.find('Conv3d') != -1):
                if classname.find('Conv3d') != -1:
                    print(m)
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    
    
    
        
    def init_SparseTransformer(self):
        depths = 4
        self.depths = depths
        num_heads = 4
        window_size = (8, 8)
        pool_size = (4, 4)
        self.fold_feat_size = (64, 64)



        kernel_size = (5, 5)
        padding = (2, 2)
        stride = (2, 2)
        t2t_params = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding
        }
        
        self.softsplit = SoftSplit(192, 512, kernel_size, stride, padding,gain=0.01)
        self.softcomp = SoftComp(192, 512, kernel_size, stride, padding,gain=0.01)
        self.transsqattn = TemporalSparseTransformerBlock(512,num_heads,window_size,pool_size,depths=depths,t2t_params=t2t_params,gain=0.01)
        self.max_pool = nn.AvgPool2d(kernel_size, stride, padding)
        # self.max_pool2 = nn.MaxPool2d((4,4), (4,4), (0,0))
        self.max_pool2 = nn.AvgPool2d(window_size, window_size, (0, 0))
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.cabattn = nn.Sequential(
            CAB2(192, 5, 4, bias=False, act=self.act, add_channel=192), 
            CAB1(192, 5, 4, bias=False, act=self.act)
        )

        
        

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def propagate(self, feats, flows, flows_check, blur_motion_map, module_name):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        last_key = list(feats)[-2]
        # print(list(feats))
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        blur_map_n1 = flows.new_zeros(n, 1, h, w)
        for i, idx in enumerate(frame_idx):
            # print(frame_idx[i])
            
            feat_current = feats[last_key][mapping_idx[idx]]
            blur_map_current = blur_motion_map[:, frame_idx[i], :, :, :]
            
            
            # print(idx)
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                flow_n1_check = flows_check[:, flow_idx[i], :, :, :]
                blur_map_n1 = blur_motion_map[:, frame_idx[i-1], :, :, :]
                # print(frame_idx[i-1])
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                blur_map_n1 = flow_warp(blur_map_n1, flow_n1.permute(0, 2, 3, 1))
                flow_n1_valid,_ = fbConsistencyCheck(flow_n1,flow_n1_check)

                
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                blur_map_n2 = torch.zeros_like(blur_map_n1)
                flow_n2_valid = torch.zeros_like(flow_n1_valid)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2_check = flows_check[:, flow_idx[i - 1], :, :, :]
                    
                    blur_map_n2 = blur_motion_map[:, frame_idx[i-2], :, :, :]
                    # print(frame_idx[i-2])
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    flow_n2_check = flow_n1_check + flow_warp(flow_n2_check, flow_n1_check.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                    blur_map_n2 = flow_warp(blur_map_n2, flow_n2.permute(0, 2, 3, 1))
                    flow_n2_valid,_ = fbConsistencyCheck(flow_n2,flow_n2_check)
                    
                # flow-guided deformable convolution
                cond = torch.cat([ cond_n1 ,feat_current, cond_n2 , blur_map_n1, blur_map_current, blur_map_n2,flow_n1_valid,flow_n2_valid], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2, blur_map_n1*flow_n1_valid, blur_map_n2*flow_n2_valid)
            
            # concatenate and residual blocks
            feat = [feats[k][idx] for k in feats if k not in [module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]
            # n d c h w
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)
            
            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
        

        return feats

    def upsample(self, lqs, feats):
        hr = feats['trans']
        hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
        hr += lqs
        return hr
        

    def forward(self, lqs,flows_forwards_gt,flows_backwards_gt):
        """Forward function for BSSTNet.

        Args:
            lqs (tensor): Input low quality sequence with
                shape (n, t, 3, 256, 256).

        Returns:
            Tensor: Output HR sequence with shape (n, t, 3, 256, 256).
        """
        # start_time = time.time()
        n, t, _, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)
        

        # shallow feature extractions
        feats = {}
        #  shitnet extract features
        x = lqs[0]
        x = self.concat(x)
        shortcut = x
        x = self.down01(x)
        enc1 = self.encoder_level1(x)
        enc11 = self.encoder_level1_1(enc1)
        enc1_down = self.down12(enc11)
        enc2 = self.encoder_level2(enc1_down)
        enc22 = self.encoder_level2_1(enc2)
        feat_ = enc22.unsqueeze(0)
        ############################################
        feats['spatial'] = [feat_[:,i,...] for i in range(feat_.size(1))]

        flows_forward, flows_backward = flows_forwards_gt, flows_backwards_gt
        pred_motion_st_blur_map = get_blur_map(flows_forwards_gt,flows_backwards_gt)
        
        flows_forward, flows_backward = self.blur_motion_refine.forward_bidirect_flow(pred_motion_st_blur_map,[flows_forward, flows_backward])
        
        blur_map_st = get_blur_map(flows_forward, flows_backward)
        blur_map_st = blur_map_st.detach()
        
        
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                    flows_check = flows_forward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)
                    flows_check = flows_backward
                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(feats, flows,flows_check,1-blur_map_st, module_name)
        
        
        
        
        feats['spatial'] = torch.stack(feats['spatial'], 1)
        feats['backward_1'] = torch.stack(feats['backward_1'], 1)
        feats['forward_1'] = torch.stack(feats['forward_1'], 1)
        feats['backward_2'] = torch.stack(feats['backward_2'], 1)
        feats['forward_2'] = torch.stack(feats['forward_2'], 1)
        

        

        

        hr = torch.cat([feats[k] for k in feats], dim=2)
        input_feats = self.reconstruction(hr)
        

        
        mask = blur_map_st.view(n,t,1, h//4, w//4)
    
        n,t,c, h, w = blur_map_st.shape
        
        blur_map_st_win = self.max_pool(blur_map_st.view(n*t,c, h, w))
        blur_map_st_win = self.max_pool2(blur_map_st_win)
        blur_map_st_win = blur_map_st_win.view(n*t,1,blur_map_st_win.shape[-2],blur_map_st_win.shape[-1])
        _,c,h,w = blur_map_st_win.shape
        blur_map_st_win = blur_map_st_win.view(n,t,c,h,w)
        # Spatial Sparse
        blur_map_st_win_hard_map = torch.where(blur_map_st_win>0.3,torch.ones_like(blur_map_st_win),torch.zeros_like(blur_map_st_win))

        # Temporal Sparse
        T_ind = [torch.arange(i, t, 2) for i in range(2)] * (self.depths // 2)
        T_ind_zero = T_ind[::-1]

        qmasks = []
        kvmasks = []
        
        for T_idxs,T_idxs_zero in zip(T_ind,T_ind_zero):
            mask = blur_map_st_win_hard_map[:,T_idxs].sum(1,keepdim=True)
            mask = mask.clip(0,1)

            
            blur_map_st_win_q_soft = blur_map_st_win
            blur_map_st_win_q_soft[:,T_idxs_zero] = blur_map_st_win_q_soft[:,T_idxs_zero]*0.
            q_idx = torch.argsort(blur_map_st_win_q_soft,1,True)
            qmask = torch.zeros_like(blur_map_st_win)
            

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            blur_map_st_win_kv_soft = (1 - blur_map_st_win)
            blur_map_st_win_kv_soft[:,T_idxs_zero] = blur_map_st_win_kv_soft[:,T_idxs_zero]*0.
            kv_idx = torch.argsort(blur_map_st_win_kv_soft,1,True) 
            kvmask = torch.zeros_like(blur_map_st_win)

            
            kvmask = kvmask.scatter_(1,kv_idx[:,:len(T_idxs)//2],1)*mask
            qmask = qmask.scatter_(1,q_idx[:,:len(T_idxs)//2],1)*mask
            qmask = rearrange(qmask, 'b t c h w -> b t h w c').contiguous()
            kvmask = rearrange(kvmask, 'b t c h w -> b t h w c').contiguous()

            qmasks.append(qmask)
            kvmasks.append(kvmask)


        
        b,t,c,h,w = input_feats.shape
        
        trans_feats = self.softsplit(input_feats.view(b*t,c,h,w),n,self.fold_feat_size)
        

        

        trans_feats = self.transsqattn(trans_feats,self.fold_feat_size,qmasks,kvmasks)

        
        trans_feats = self.softcomp(trans_feats,t,self.fold_feat_size)

        
        trans_feats = trans_feats.view(b,t,c,h,w)
        trans_feats = self.cabattn(torch.cat([input_feats,trans_feats],2)[0]).unsqueeze(0)
        
        x = self.up21(trans_feats[0], self.skip_attn1(enc11))
        dec1 = self.decoder_level1(x)
        dec11 = self.decoder_level1_1(dec1, reverse=1)
        
        dec11_out = self.conv_hr0(self.act(self.upsample0(dec11))) + self.skip_conv(shortcut)
        dec11_out = self.out_conv(dec11_out)
        dec11_out = self.last_conv(dec11_out)
        dec11_out = dec11_out.unsqueeze(0) + lqs

        
        # reconstruction
        return dec11_out
