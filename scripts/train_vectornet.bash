# bash script to record the training config

# for test the program on the complete dataset
# python train_vectornet.py -d dataset/interm_data -o run/vectornet/ -a -b 128 -c -cd 1 --lr 0.001 -luf 10 -ldr 0.1

# for test the program on the small dataset
#python train_vectornet.py -d dataset/interm_data_small -o run/vectornet/ -a -b 128 -c -cd 1 --lr 0.001 -luf 10 -ldr 0.1

# for multi-gpu training
# nproc_per_node: set the number of gpu
#python -m torch.distributed.launch --nproc_per_node=2 train_vectornet.py -d dataset/interm_data -o run/tnt/ -a -b 256 -c -m --lr 0.0010 -luf 10 -ldr 0.5

# when you need to choose the training gpus, use this one
CUDA_VISIBLE_DEVICES=1,0 python -m torch.distributed.launch --nproc_per_node=2 train_vectornet.py -d dataset/interm_data -o run/vectornet/ -a -b 256 -c -m --lr 0.0010 -luf 10 -ldr 0.5
