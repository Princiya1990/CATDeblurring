#python3.6 train.py --dataroot /running/CUHK_CV/face2sketch --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --gpu_ids -1
#python3.6 test.py --dataroot /running/CUHK_CV/face2sketch --name facades_pix2pix --model pix2pix --direction BtoA --gpu_ids -1

python3.6 test.py --dataroot /running/Test/face2sketch --name facades_pix2pix --model pix2pix --direction BtoA --gpu_ids -1

