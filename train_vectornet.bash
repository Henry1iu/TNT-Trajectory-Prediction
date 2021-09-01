# bash script to record the training config
python train_vectornet.py -d dataset/interm_tnt_n_s_0804 -o run/vectornet/ -a -b 128 -c -cd 1 --lr 0.001 -luf 10 -ldr 0.1
# python train_vectornet.py -d dataset/interm_data -o run/vectornet/ -a -b 256 -c -cd 1 0 --lr 0.001
