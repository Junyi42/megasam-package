
# mkdir -p output_folder
# ffmpeg -i video.mp4 -vf "fps=30" output_folder/frame_%04d.png

evalset=(
  parkour
)

DATA_DIR=DAVIS
GPU=3

# Run DepthAnything
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path $DATA_DIR/$seq \
  --outdir Depth-Anything/video_visualization/$seq
done

# Run UniDepth
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python UniDepth/scripts/demo_mega-sam.py \
  --scene-name $seq \
  --img-path $DATA_DIR/$seq \
  --outdir UniDepth/outputs
done

DATA_PATH=$DATA_DIR
CKPT_PATH=checkpoints/megasam_final.pth


for seq in ${evalset[@]}; do
    CUDA_VISIBLE_DEVICE=$GPU python camera_tracking_scripts/test_demo.py \
    --datapath=$DATA_PATH/$seq \
    --weights=$CKPT_PATH \
    --scene_name $seq \
    --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
    --metric_depth_path $(pwd)/UniDepth/outputs \
    --disable_vis $@
done

# Run Raft Optical Flows
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_PATH/$seq \
  --model=cvd_opt/raft-things.pth \
  --scene_name $seq --mixed_precision
done

# Run CVD optmization
for seq in ${evalset[@]}; do
  CUDA_VISIBLE_DEVICES=$GPU python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0
done