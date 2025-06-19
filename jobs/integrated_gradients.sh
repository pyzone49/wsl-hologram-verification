python integrated_gradients.py \
--config-name=wsl +experiment=wsl/mobilevit_s    \
 "paths.split_name=k0" "decision=cumulative"     \
 +image_dir="/home/diva/Documents/other/pouliquen.24.icdar/data/midv-holo/crop_ovds/origins/ID/id01_05_01" \
 +checkpoint_path="/home/diva/Documents/other/pouliquen.24.icdar/checkpoints/backbones/backbone_epoch_99.pth"
