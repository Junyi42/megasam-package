#!/bin/bash

# bash run_all.sh --video_path "video.mp4" --gpu 0 --video_name "video"

# Default GPU value
GPU=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video_path)
      video_path="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --video_name)
      video_name="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      shift 1
      ;;
  esac
done

# If video_name is not set, use basename of video_path
if [ -z "$video_name" ]; then
  video_name=$(basename "$video_path")
fi

# Check if video_path is set
if [ -z "$video_path" ]; then
  echo "Error: video_path must be specified"
  exit 1
fi

# If video_path is a folder, move it to the DAVIS directory and rename it if necessary
if [ -d "$video_path" ]; then
  # Create DAVIS folder if it doesn't exist
  mkdir -p "DAVIS"
  
  # If video_name is provided, rename the folder
  if [ -n "$video_name" ]; then
    cp -r "$video_path" "DAVIS/$video_name"
  else
    cp -r "$video_path" "DAVIS/"
  fi
else
  # If it's a video file, run ffmpeg to extract frames
  mkdir -p "DAVIS/$video_name"
  ffmpeg -i "$video_path" -vf "fps=30" "DAVIS/$video_name/frame_%04d.png"
fi

evalset=(
  $video_name
)

DATA_DIR=DAVIS
CKPT_PATH=checkpoints/megasam_final.pth

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

# Run DepthAnything
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path $DATA_DIR/$seq \
  --outdir Depth-Anything/video_visualization/$seq
done

for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python UniDepth/scripts/demo_mega-sam.py \
  --scene-name $seq \
  --img-path $DATA_DIR/$seq \
  --outdir UniDepth/outputs
done

for seq in ${evalset[@]}; do
    CUDA_VISIBLE_DEVICES=$GPU python camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_DIR/$seq \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis $@ 
done

# Run Raft Optical Flows
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_DIR/$seq \
  --model=cvd_opt/raft-things.pth \
  --scene_name $seq --mixed_precision
done

# Run CVD optimization
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done
