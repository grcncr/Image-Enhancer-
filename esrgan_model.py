"""Real-ESRGAN model implementation without basicsr dependency.

This implements the RRDBNet architecture and inference pipeline directly,
allowing it to work on any Python version with just PyTorch.
"""

import logging
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --- RRDBNet Architecture ---


def make_layer(block, n_layers, **kwargs):
    layers = [block(**kwargs) for _ in range(n_layers)]
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling (named layers to match pretrained weights)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # 2x upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        # 2x upsample (total 4x)
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# --- Inference Pipeline ---


class RealESRGANUpscaler:
    """Real-ESRGAN upscaler with tiled inference for large images."""

    def __init__(self, model_path: str, device: str = "cpu", tile_size: int = 512, tile_pad: int = 10):
        self.device = torch.device(device)
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = 4

        # Build and load model
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
        loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if "params_ema" in loadnet:
            keyname = "params_ema"
        elif "params" in loadnet:
            keyname = "params"
        else:
            keyname = None

        if keyname:
            self.model.load_state_dict(loadnet[keyname], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)

        self.model.eval()
        self.model = self.model.to(self.device)
        logger.info(f"Real-ESRGAN model loaded on {device}")

    @torch.no_grad()
    def enhance(self, img_rgb: np.ndarray, outscale: float = 4.0) -> np.ndarray:
        """Enhance an image (RGB uint8 input, RGB uint8 output)."""
        # Normalize to [0, 1] float32
        img = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        h, w = img.shape[:2]

        # If image is small enough, process directly
        if h * w <= self.tile_size * self.tile_size:
            output = self.model(img_tensor)
        else:
            output = self._tile_process(img_tensor)

        # Convert back to numpy uint8
        output = output.squeeze(0).cpu().clamp(0, 1).numpy()
        output = (output.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

        # If outscale != 4, resize to target
        if outscale != 4.0:
            target_h = int(h * outscale)
            target_w = int(w * outscale)
            output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return output

    def _tile_process(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Process image in tiles to handle large images without OOM."""
        batch, channel, height, width = img_tensor.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output = img_tensor.new_zeros((batch, channel, output_height, output_width))

        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Input tile area
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size

                # With padding
                input_start_x = max(ofs_x - self.tile_pad, 0)
                input_end_x = min(ofs_x + self.tile_size + self.tile_pad, width)
                input_start_y = max(ofs_y - self.tile_pad, 0)
                input_end_y = min(ofs_y + self.tile_size + self.tile_pad, height)

                # Process tile
                input_tile = img_tensor[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                output_tile = self.model(input_tile)

                # Output tile area
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # Remove padding from output
                output_start_x_tile = (ofs_x - input_start_x) * self.scale
                output_end_x_tile = output_start_x_tile + min(self.tile_size, width - ofs_x) * self.scale
                output_start_y_tile = (ofs_y - input_start_y) * self.scale
                output_end_y_tile = output_start_y_tile + min(self.tile_size, height - ofs_y) * self.scale

                # Place in output
                out_x_start = ofs_x * self.scale
                out_x_end = out_x_start + (output_end_x_tile - output_start_x_tile)
                out_y_start = ofs_y * self.scale
                out_y_end = out_y_start + (output_end_y_tile - output_start_y_tile)

                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

        return output
