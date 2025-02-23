# CUDA_VISIBLE_DEVICES=1 python run_all.py --img_path DAVIS --iterate --delta

import os
import sys
import glob
import argparse
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import cv2
import imageio
from PIL import Image
from timeit import default_timer as timer
from tqdm import tqdm
from torchvision.transforms import Compose

# Append necessary directories to sys.path
sys.path.append('Depth-Anything')
sys.path.append('UniDepth')
sys.path.append("base/droid_slam")
sys.path.append('cvd_opt/core')
sys.path.append('cvd_opt')
sys.path.append("base/droid_slam")
sys.path.append("../DELTA_densetrack3d")

from raft import RAFT
from droid import Droid
from core.utils.utils import InputPadder
from pathlib import Path  # pylint: disable=g-importing-member
from lietorch import SE3
from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid
from camera_tracking_scripts.test_demo import droid_slam_optimize, return_full_reconstruction
from preprocess_flow import prepare_img_data, process_flow
from cvd_opt import cvd_optimize

from densetrack3d.datasets.custom_data import read_data_with_megasam
from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from transfer_to_world import process_3d_tracking

LONG_DIM = 640

def demo_unidepth(model, img_path_list, args, save=False):
  outdir = args.outdir  # "./outputs"
  # os.makedirs(outdir, exist_ok=True)

  # for scene_name in scene_names:
  scene_name = args.scene_name
  outdir_scene = os.path.join(outdir, scene_name)
  os.makedirs(outdir_scene, exist_ok=True)

  fovs = []
  depth_list = []
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
    # print(fov_)
    fovs.append(fov_)
    depth_list.append(np.float32(depth))
    # breakpoint()
    if save:
        np.savez(
            os.path.join(outdir_scene, img_path.split("/")[-1][:-4] + ".npz"),
            depth=np.float32(depth),
            fov=fov_,
        )
  return depth_list, fovs  

def demo_depthanything(depth_anything, filenames, args, save=False):
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

    if save:
        os.makedirs(os.path.join(args.outdir), exist_ok=True)
        np.save(
            os.path.join(args.outdir, filename.split('/')[-1][:-4] + '.npy'),
            depth_npy,
        )
    final_results.append(depth_npy)
  return final_results

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_path', type=str)
  parser.add_argument('--outdir', type=str, default='./vis_depth')
  parser.add_argument("--scene-name", type=str, default=None)
  parser.add_argument("--save_intermediate", action="store_true", default=False)
  parser.add_argument("--iterate", action="store_true", default=False)
  parser.add_argument("--num_frames", type=int, default=512)
  parser.add_argument("--stride", type=int, default=1)

  # for unidepth & depthanything
  parser.add_argument('--encoder', type=str, default='vitl')
  parser.add_argument('--load-from', type=str, default="Depth-Anything/checkpoints/depth_anything_vitl14.pth")
  # parser.add_argument('--max_size', type=int, required=True)
  parser.add_argument('--localhub', dest='localhub', action='store_true', default=False)
  
  # for raft
  parser.add_argument('--model', default='cvd_opt/raft-things.pth', help='restore checkpoint')
  parser.add_argument('--mixed_precision', type=bool, default=True, help='use mixed precision')
  parser.add_argument('--num_heads',default=1,type=int,help='number of heads in attention and aggregation')
  parser.add_argument('--position_only',default=False,action='store_true',help='only use position-wise attention')
  parser.add_argument('--position_and_content',default=False,action='store_true',help='use position and content-wise attention')
  parser.add_argument('--small', action='store_true', help='use small model')

  # for cvd optimize
  parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
  parser.add_argument("--w_normal", type=float, default=5.0, help="w_normal")

  # for droid slam
  parser.add_argument("--weights", default="checkpoints/megasam_final.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument(
      "--filter_thresh", type=float, default=2.0
  )  # motion threhold for keyframe
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--scene_name", help="scene_name")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)
  parser.add_argument("--disable_vis", type=bool, default=True)

  parser.add_argument("--delta", action="store_true")
  parser.add_argument("--delta_ckpt", type=str, default="DELTA_densetrack3d/checkpoints/densetrack3d.pth", help="checkpoint path")
  parser.add_argument("--upsample_factor", type=int, default=4, help="model stride")
  parser.add_argument("--use_fp16", action="store_true", help="whether to use fp16 precision")
  parser.add_argument("--save_world_tracks", action="store_true", help="whether to save world tracks")

  args = parser.parse_args()
  out_dir = args.outdir
  scene_name = args.scene_name if args.scene_name else out_dir.split('/')[-1]
  w_grad = args.w_grad
  w_normal = args.w_normal

  # step 1: prepare unidepth

  model_uni = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_uni = model_uni.to(device)


  # step 2: prepare depth anything

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

  # step 3: prepare the flow model

  model_raft = torch.nn.DataParallel(RAFT(args))
  model_raft.load_state_dict(torch.load(args.model))
  print(f'Loaded checkpoint at {args.model}')
  flow_model = model_raft.module
  flow_model.cuda()  # .eval()
  flow_model.eval()

  # step 4: delta densetrack3d

  if args.delta:
    delta_model = DenseTrack3D(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        upsample_factor=args.upsample_factor
    )
    with open(args.delta_ckpt, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    delta_model.load_state_dict(state_dict, strict=False)

    predictor = DensePredictor3D(model=delta_model)
    predictor = predictor.eval().cuda()

  # run the pipeline

  if args.iterate: # get all the subfolders in the args.img_path
    folders = sorted([os.path.join(args.img_path, f) for f in os.listdir(args.img_path) if os.path.isdir(os.path.join(args.img_path, f))])
    # folders = folders[:16]
  else:
    folders = [args.img_path]
  print(f"Processing {len(folders)} folders")
  for img_path in tqdm(folders):
    scene_name = img_path.split("/")[-1]
    save_path = os.path.join(out_dir, scene_name)
    os.makedirs(save_path, exist_ok=True)

    # step1&2&3: Run the demo
    img_path_list = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
    img_path_list += sorted(glob.glob(os.path.join(img_path, "*.png")))
    img_path_list = img_path_list[:args.num_frames]

    depth_list_uni, fovs = demo_unidepth(model_uni, img_path_list, args, save=args.save_intermediate)
    depth_list_da = demo_depthanything(depth_anything, img_path_list, args, save=args.save_intermediate)
    img_data = prepare_img_data(img_path_list)
    flows_high, flow_masks_high, iijj = process_flow(flow_model, img_data, args.scene_name if args.save_intermediate else None)
    
    # step 4: Run the droid slam
    droid, traj_est, rgb_list, senor_depth_list, motion_prob = droid_slam_optimize(
        img_path_list, depth_list_da, depth_list_uni, fovs, args
    )

    images, disps, poses, intrinsics, motion_prob = return_full_reconstruction(
            droid, traj_est, rgb_list, senor_depth_list, motion_prob
        )

    # step 5: Run the cvd optimize
    images, depths, intrinsics, cam_c2w = cvd_optimize(
            images[:, ::-1, ...],
            disps + 1e-6,
            poses,
            intrinsics,
            motion_prob,
            flows_high,
            flow_masks_high,
            iijj,
            out_dir,
            scene_name,
            w_grad,
            w_normal
        )
    
    print(f"Megasam Finished processing {scene_name}, saved to {out_dir}/{scene_name}")

    # step 6: Run the delta densetrack3d
    if args.delta:
      video = torch.from_numpy(images).permute(0, 3, 1, 2).cuda()[None].float()
      videodepth = torch.from_numpy(depths).unsqueeze(1).cuda()[None].float()
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_fp16):
        out_dict = predictor(
            video,
            videodepth,
            grid_query_frame=0,
            save_color=args.save_world_tracks,
        )
      with open(os.path.join(save_path, "delta_results.pkl"), "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Delta Finished processing {scene_name}, saved to {out_dir}/{scene_name}")

    # step 7: Run the transfer to world
    if args.save_world_tracks:
      process_3d_tracking(
          out_dict["colors"][0],
          out_dict["trajs_uv"][0],
          out_dict["trajs_depth"][0],
          intrinsics,
          cam_c2w,
          save_path = os.path.join(save_path, f"dense_3d_track_world.pkl")
      )
      print(f"World tracks saved to {out_dir}/{scene_name}")