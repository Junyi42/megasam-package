import argparse
import glob
import os
import sys
sys.path.append('Depth-Anything')
sys.path.append('UniDepth')
# import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cv2
from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import argparse
import glob
import os

import cv2
import imageio
import numpy as np
from PIL import Image
import torch
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid

LONG_DIM = 640

def demo_unidepth(model, img_path_list, args):
  outdir = args.outdir  # "./outputs"
  # os.makedirs(outdir, exist_ok=True)

  # for scene_name in scene_names:
  scene_name = args.scene_name
  outdir_scene = os.path.join(outdir, scene_name)
  os.makedirs(outdir_scene, exist_ok=True)

  fovs = []
  for img_path in tqdm(img_path_list):
    rgb = np.array(Image.open(img_path))[..., :3]
    if rgb.shape[1] > rgb.shape[0]:
      final_w, final_h = LONG_DIM, int(
          round(LONG_DIM * rgb.shape[0] / rgb.shape[1])
      )
    else:
      final_w, final_h = (
          int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])),
          LONG_DIM,
      )
    rgb = cv2.resize(
        rgb, (final_w, final_h), cv2.INTER_AREA
    )  # .transpose(2, 0, 1)

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    # predict
    predictions = model.infer(rgb_torch)
    fov_ = np.rad2deg(
        2
        * np.arctan(
            predictions["depth"].shape[-1]
            / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
        )
    )
    depth = predictions["depth"][0, 0].cpu().numpy()
    print(fov_)
    fovs.append(fov_)
    # breakpoint()
    np.savez(
        os.path.join(outdir_scene, img_path.split("/")[-1][:-4] + ".npz"),
        depth=np.float32(depth),
        fov=fov_,
    )

def demo_depthanything(depth_anything, filenames, args):
  transform = Compose([
      Resize(
          width=768,
          height=768,
          resize_target=False,
          keep_aspect_ratio=True,
          ensure_multiple_of=14,
          resize_method='upper_bound',
          image_interpolation_method=cv2.INTER_CUBIC,
      ),
      NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      PrepareForNet(),
  ])

  final_results = []
  for filename in tqdm(filenames):
    raw_image = cv2.imread(filename)[..., :3]
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).cuda()

    # start = timer()
    with torch.no_grad():
      depth = depth_anything(image)
    # end = timer()

    depth = F.interpolate(
        depth[None], (h, w), mode='bilinear', align_corners=False
    )[0, 0]
    depth_npy = np.float32(depth.cpu().numpy())
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)

    os.makedirs(os.path.join(args.outdir), exist_ok=True)
    np.save(
        os.path.join(args.outdir, filename.split('/')[-1][:-4] + '.npy'),
        depth_npy,
    )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img-path', type=str)
  parser.add_argument('--outdir', type=str, default='./vis_depth')

  parser.add_argument('--encoder', type=str, default='vitl')
  parser.add_argument('--load-from', type=str, default="Depth-Anything/checkpoints/depth_anything_vitl14.pth")
  # parser.add_argument('--max_size', type=int, required=True)

  parser.add_argument(
      '--localhub', dest='localhub', action='store_true', default=False
  )

  parser.add_argument("--scene-name", type=str, default="scene_name")

  args = parser.parse_args()

  img_path_list = sorted(glob.glob(os.path.join(args.img_path, "*.jpg")))
  img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.png")))

  # step 1: unidepth
  
  model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  demo_unidepth(model, img_path_list, args)


  # step 2: depth anything

  assert args.encoder in ['vits', 'vitb', 'vitl']
  if args.encoder == 'vits':
    depth_anything = DPT_DINOv2(
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384],
        localhub=args.localhub,
    ).cuda()
  elif args.encoder == 'vitb':
    depth_anything = DPT_DINOv2(
        encoder='vitb',
        features=128,
        out_channels=[96, 192, 384, 768],
        localhub=args.localhub,
    ).cuda()
  else:
    depth_anything = DPT_DINOv2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        localhub=args.localhub,
    ).cuda()

  total_params = sum(param.numel() for param in depth_anything.parameters())
  print('Total parameters: {:.2f}M'.format(total_params / 1e6))

  depth_anything.load_state_dict(
      torch.load(args.load_from, map_location='cpu'), strict=True
  )

  depth_anything.eval()
  
  demo_depthanything(depth_anything, img_path_list, args)
