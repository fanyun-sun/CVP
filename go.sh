#!/bin/bash -ex
python3 write_train_test_txt.py
python3 train.py --dataset vvn/collide2_new --gpu 3,4
#python3 train.py --dataset vvn/slide_roll --gpu 3,4
#python3 mytest.py --dataset vvn --gpu 1 --checkpoint ./output/tmp/vvnrgb_cvp_traj_comb_late_fact_gc_n2e2n_Ppix1_iter100000.pth
