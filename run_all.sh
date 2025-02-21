
video_path="ego4d1c4.mp4"
video_name=$(basename "$video_path")
GPU=0

mkdir -p "DAVIS/$video_name"
ffmpeg -i "$video_path" -vf "fps=30" "DAVIS/$video_name/frame_%04d.png"


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
    CUDA_VISIBLE_DEVICE=$GPU python camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_DIR/$seq \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis $@ \
    --gpu $GPU
done

# Run Raft Optical Flows
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_DIR/$seq \
  --model=cvd_opt/raft-things.pth \
  --scene_name $seq --mixed_precision
done

# Run CVD optmization
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done