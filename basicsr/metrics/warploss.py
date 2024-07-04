import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.archs.arch_util import flow_warp
from basicsr.losses.basic_loss import l1_loss




@METRIC_REGISTRY.register()
def WarpLoss(gt_frames, flow_backwards, flow_forwards, weight=None, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    n,t,c,h,w = gt_frames.shape
    lqs_1 = gt_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
    lqs_2 = gt_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
    lqs_1 = F.adaptive_avg_pool2d(lqs_1,(h//4,w//4))
    lqs_2 = F.adaptive_avg_pool2d(lqs_2,(h//4,w//4))
    lqs_1_warp_2 = flow_warp(lqs_1,flow_forwards.reshape(-1,2,h//4,w//4).permute(0, 2, 3, 1))
    lqs_2_warp_1 = flow_warp(lqs_2,flow_backwards.reshape(-1,2,h//4,w//4).permute(0, 2, 3, 1))
    flow_forwards_loss = l1_loss(lqs_1_warp_2, lqs_2, weight, reduction="mean")
    flow_backwards_loss = l1_loss(lqs_2_warp_1, lqs_1, weight, reduction="mean")
    flow_loss = 0.5*flow_forwards_loss + 0.5*flow_backwards_loss

    return flow_loss

