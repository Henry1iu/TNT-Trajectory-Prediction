# bash script to record the training config

# for test the program on the complete dataset

# python train_tnt.py -d dataset/interm_data -o run/tnt/ -a -b 64 -c -cd 1 0 --lr 0.0010 -luf 10 -ldr 0.1

# for test the program on the small dataset
python train_tnt.py -d dataset/interm_data_small -o run/tnt/ -a -b 64 -c -cd 1 0 --lr 0.0010 -luf 10 -ldr 0.1
