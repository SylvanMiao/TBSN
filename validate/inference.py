# Single-image denoising inference for TBSN / AP-BSN.
#
# Usage:
#   python validate/inference.py \
#       --config option/tbsn_confocal.json \
#       --checkpoint path/to/checkpoint.pth \
#       --input path/to/noisy.tif \
#       --output path/to/denoised.tif \
#       [--tile N]
#
#   --tile 0  : full-image inference (default).
#                Matches APBSNModel.validation_step exactly.
#                Use when image resolution matches training patch_size
#                (e.g. 512x512 image + patch_size=512 model).
#
#   --tile 64 : tiled inference with NxN tiles, no overlap, direct stitch.
#                Use when the model was trained at a smaller patch_size
#                than the input image (e.g. 1024x1024 image + patch_size=64 model).
#                Tile size must evenly divide image dimensions.
#
# Or via the shell wrapper:
#   sh denoise.sh <checkpoint> <input> <output> [config] [tile]

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from model.apbsn import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from util.option import parse


def r3_refine(net, noisy_input, initial_denoised, R3_T, R3_p):
    """R3 (Random Replacement Refinement): run T iterations, each time replacing
    random pixels with the original noisy input, feed through the network, and
    average all results."""
    b, c, h, w = noisy_input.shape
    denoised = torch.empty(b, c, h, w, R3_T, device=noisy_input.device)
    for t in range(R3_T):
        indice = torch.rand_like(noisy_input)
        mask = indice < R3_p
        tmp_input = torch.clone(initial_denoised).detach()
        tmp_input[mask] = noisy_input[mask]
        with torch.no_grad():
            tmp_output = net(tmp_input)[:, :, :h, :w]
        denoised[..., t] = tmp_output
    return torch.mean(denoised, dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Single-image denoising inference")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--input', type=str, required=True, help='Path to noisy input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save denoised output image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint .pth')
    parser.add_argument('--tile', type=int, default=0, help='Tile size for tiled inference (0 = full-image, no tiling)')
    args = parser.parse_args()

    opt = parse(args.config)

    # ---- build network ----
    Net = getattr(__import__('network'), opt['networks'][0]['type'])
    net = build_net(Net, opt['networks'][0]['args'])
    load_checkpoint(net, args.checkpoint)
    net = net.cuda()
    net.eval()

    pd_b = opt.get('pd_b', 2)
    R3 = opt.get('R3', False)
    R3_T = opt.get('R3_T', 8)
    R3_p = opt.get('R3_p', 0.16)
    if R3:
        print(f'R3 refinement enabled: T={R3_T}, p={R3_p}')

    # ---- load & preprocess image ----
    img = Image.open(args.input)
    img_np = np.asarray(img)
    if img_np.ndim == 3:
        img_np = img_np[:, :, 0]
    if img_np.dtype == np.uint16:
        img_np = img_np.astype(np.float32) / 65535.
    elif img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.
    else:
        img_np = img_np.astype(np.float32)
        if img_np.max() > 1:
            img_np /= 255.

    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, H, W)
    input_t = tensor * 255.
    _, _, H, W = input_t.shape

    if args.tile > 0:
        # ---- tiled inference ----
        tile_size = args.tile
        assert H % tile_size == 0 and W % tile_size == 0, \
            f'Image ({H}x{W}) must be divisible by tile size ({tile_size})'

        n_h = H // tile_size
        n_w = W // tile_size
        print(f'Tiled inference: {n_h}x{n_w} = {n_h * n_w} tiles '
              f'(tile={tile_size}, image={H}x{W})')

        tiles_out = []
        for iy in tqdm(range(n_h), desc='Tiles (rows)'):
            row_tiles = []
            for ix in range(n_w):
                y, x = iy * tile_size, ix * tile_size
                tile = input_t[:, :, y:y+tile_size, x:x+tile_size]

                tile_pd = pixel_shuffle_down_sampling(tile, f=pd_b)
                with torch.no_grad():
                    tile_out_pd = net(tile_pd)
                tile_out = pixel_shuffle_up_sampling(tile_out_pd, f=pd_b)[:, :, :tile_size, :tile_size]

                if R3:
                    tile_out = r3_refine(net, tile, tile_out, R3_T, R3_p)

                row_tiles.append(tile_out)
            tiles_out.append(torch.cat(row_tiles, dim=3))

        denoised = torch.cat(tiles_out, dim=2)[:, :, :H, :W] / 255.
    else:
        # ---- full-image inference (matching APBSNModel.validation_step) ----
        print(f'Full-image inference: image={H}x{W}')
        b, c, h, w = input_t.shape
        input_pd = pixel_shuffle_down_sampling(input_t, f=pd_b)
        with torch.no_grad():
            output_pd = net(input_pd)
        output = pixel_shuffle_up_sampling(output_pd, f=pd_b)[:, :, :h, :w]

        if R3:
            output = r3_refine(net, input_t, output, R3_T, R3_p)

        denoised = output / 255.

    # ---- save ----
    result = denoised.squeeze().cpu().numpy()
    result = np.clip(result, 0., 1.)
    result = (result * 65535.).astype(np.uint16)
    Image.fromarray(result).save(args.output)
    print(f'Denoised image saved to: {args.output}')


def build_net(net_class, net_args):
    return net_class(**net_args)


def load_checkpoint(net, path):
    state_dict = torch.load(path, map_location='cpu')
    if 'model_weight' in state_dict:
        state_dict = state_dict['model_weight']['denoiser']
    keys = list(state_dict.keys())
    if keys and keys[0].startswith('bsn.'):
        state_dict = {k.replace('bsn.', ''): v for k, v in state_dict.items()}
    if keys and keys[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    print(f'Loaded checkpoint: {path}')


if __name__ == '__main__':
    main()
