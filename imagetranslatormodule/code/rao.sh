python3.6 train.py --dataroot /running/CUHK_CV/face2sketch --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids 0,1
