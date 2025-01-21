import os
os.chdir("code")
command = "python train.py --dataroot --dataroot ./../data/CUHK --model pix2pix --direction BtoA"
# Execute the command
os.system(command)
