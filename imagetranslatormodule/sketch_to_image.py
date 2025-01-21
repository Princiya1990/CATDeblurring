import os
os.chdir("code")
combine="python combine_A_and_B.py --fold_A ./../data/CUHK/photos --fold_B ./../data/CUHK/photos --fold_AB ./../data/CUHK/face2sketch"
os.system(combine)
command = "python train.py --dataroot ./../data/CUHK --model pix2pix --direction BtoA"
# Execute the command
os.system(command)
