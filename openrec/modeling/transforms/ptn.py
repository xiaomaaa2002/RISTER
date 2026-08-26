import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from openrec.modeling.common import Block
from .moran import MORN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def scale_inter(image: torch.Tensor, mask: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Batch 行内插值填充，image: (B,C,H,W)，mask: (B,1,H,W)
    """
    
    B, C, H, W = image.shape
    img = image.float()
    m = mask.float().squeeze(1)  # (B,H,W)

    # -------------------------
    # 按行计算累积分布
    # -------------------------
    cdf = torch.cumsum(m, dim=2)  # (B,H,W)
    row_sum = cdf[:, :, -1:].clamp_min(eps)
    cdf = cdf / row_sum

    # target
    tgt = torch.linspace(0, 1, W, device=img.device).view(1, 1, W).expand(B, H, W)
    idx = torch.searchsorted(cdf, tgt.contiguous(), right=True)
    idx = idx.clamp(1, W-1)

    idx0 = idx - 1
    idx1 = idx
    cdf0 = torch.gather(cdf, 2, idx0)
    cdf1 = torch.gather(cdf, 2, idx1)
    alpha = (tgt - cdf0) / (cdf1 - cdf0 + eps)  # (B,H,W)

    # gather
    idx0c = idx0.unsqueeze(1).expand(-1, C, -1, -1)
    idx1c = idx1.unsqueeze(1).expand(-1, C, -1, -1)
    img_perm = img  # already (B,C,H,W)
    v0 = torch.gather(img_perm, 3, idx0c)
    v1 = torch.gather(img_perm, 3, idx1c)

    out = (1 - alpha.unsqueeze(1)) * v0 + alpha.unsqueeze(1) * v1
    return out

class PolarTransformer(nn.Module):
    def __init__(self, in_channels=3, target_height=224, target_width=224):
        super(PolarTransformer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.out_channels = in_channels

    def _interpolate(self, im, x, y):
        im = im.permute(0, 2, 3, 1)

        B, H, W, C = im.shape
        height_f = float(H)
        width_f = float(W)
        out_height, out_width = self.target_height, self.target_width
        zero = torch.zeros([], dtype=torch.int32)
        max_y = H - 1
        max_x = W - 1

        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0

        x0 = torch.floor(x).to(torch.int32)
        x1 = x0 + 1
        y0 = torch.floor(y).to(torch.int32)
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        im_flat = im.reshape(-1, C).float()

        dim2 = W
        dim1 = W * H
        base = (torch.arange(B, dtype=torch.int32) * dim1).repeat_interleave(out_height * out_width).to(im.device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output.view(B, out_height, out_width, C)

    def forward(self, batch_image, mask):

        batch_size, channel, width, height = batch_image.shape
        
        center = self.target_width // 2 

        centers = torch.tensor([[center],[center]]).repeat(batch_size, 1, 1).squeeze(-1).to(batch_image.device)

        # if self.training:
        #     shift = 1. / channel * (torch.rand([batch_size, 2], device=batch_image.device) * 2 * 4 - 4)
        #     centers += shift

        center_x, center_y = centers[:, 0].unsqueeze(-1).unsqueeze(-1), centers[:, 1].unsqueeze(-1).unsqueeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.flip(torch.linspace(-1, 1, self.target_height), dims=[0]).to(batch_image.device),
            torch.linspace(-1, 1, self.target_width).to(batch_image.device)
        )

        max_radius = torch.tensor(0.5 * math.sqrt((height)**2 + (width)**2)).to(batch_image.device)

        grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
        grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()

        r_s = ((torch.exp((grid_x + 1) / 2 * torch.log(max_radius))) - 1) * (max_radius / (max_radius - 1))
        # r_s = (grid_x + 1) / 2 * (max_radius-1) * (max_radius / (max_radius - 1))
        r_s = (r_s) / (self.target_width) * 2 * max_radius / self.target_width
        t_s = (grid_y + 1) * torch.pi

        r = r_s * torch.cos(t_s) * max_radius + center_x
        theta = r_s * torch.sin(t_s) * max_radius + center_y

        r = (r - center) / center
        theta = (theta - center) / center 

        polar_image = self._interpolate(batch_image, r.view(-1), theta.view(-1)).permute(0, 3, 1, 2)
    
        # if mask is not None:
        #     polar_mask = self._interpolate(mask, r.view(-1), theta.view(-1)).permute(0, 3, 1, 2)
        #     # from PIL import Image
        #     # from torchvision import transforms
        #     # toPIL = transforms.ToPILImage()
        #     # for i in range(32):
        #     #     img = mask[i]
        #     #     # img = img * 0.5 + 0.5      # [-1,1] → [0,1]
        #     #     # img = img.clamp(0, 1)      # 防止数值越界
        #     #     pic = toPIL(img)
        #     #     pic.save('img/maskpic'+str(i)+'.jpg')
        #     #     img = polar_mask[i]
        #     #     # img = img * 0.5 + 0.5      # [-1,1] → [0,1]
        #     #     # img = img.clamp(0, 1)      # 防止数值越界
        #     #     pic = toPIL(img)
        #     #     pic.save('img/maskpic_polar'+str(i)+'.jpg')
        #     #     img = polar_image[i]
        #     #     img = img * 0.5 + 0.5      # [-1,1] → [0,1]
        #     #     img = img.clamp(0, 1)      # 防止数值越界
        #     #     pic = toPIL(img)
        #     #     pic.save('img/tpspic'+str(i)+'.jpg')
        #     # # exit()
        #     polar_image = scale_inter(polar_image, polar_mask)
        
            # return polar_image
        # else:
        return polar_image
