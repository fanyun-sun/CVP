#!/bin/bash -ex
#python3 train.py --dataset vvn --gpu 5
python3 mytest.py --dataset vvn --gpu 5 --checkpoint ./output/tmp/vvnrgb_cvp_traj_comb_late_fact_gc_n2e2n_Ppix1_iter108.pth
