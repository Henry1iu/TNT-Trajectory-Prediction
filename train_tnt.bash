# bash script to record the training config
#python train_tnt.py -d dataset/interm_tnt_with_filter -o run/tnt/ -a -b 256 -c -cd 1 0 --lr 0.003
#python train_tnt.py -d dataset/interm_tnt_n_s -o run/tnt/ -a -b 128 -c -cd 1 0 --lr 0.0007
#python train_tnt.py -d dataset/interm_tnt_n_s -o run/tnt/ -a -b 128 -c -cd 1 0 --lr 0.001
#python train_tnt.py -d dataset/interm_tnt_n_s -o run/tnt/ -a -b 128 -c -cd 1 0 --lr 0.003

python train_tnt.py -d dataset/interm_tnt_n_s -o run/tnt/ -a -b 128 -c -cd 1 0 --lr 0.001

#python train_tnt.py -d dataset/interm_tnt_n_s -o run/tnt/ -a -b 256 --lr 0.003