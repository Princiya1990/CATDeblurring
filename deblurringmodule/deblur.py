import os
combine="python combine_A_and_B.py --fold_A ./datasets/blurred --fold_B ./datasets/clear --fold_AB ./datasets/blur2clear"
os.system(combine)
train = "python train.py --dataroot ./datasets/blur2clear --learn_residual --resize_or_crop 'scale_width'"
os.system(train)
test = "python /content/DeblurGAN/test.py --dataroot ./datasets/test --model test --dataset_mode single --learn_residual --resize_or_crop 'scale_width'"
# Execute the command
os.system(test)
