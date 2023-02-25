# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:25:58 2022

@author: haoyu
"""

import os

noise_ratio = [0.1, 0.2, 0.4, 0.6]
# noise_ratio = [0.4]
# noise_modes = ['sym']
num_samples = [1, 2, 5, 10, 20]
noise_modes = ['asym', 'sym']

for subid in range(1, 15):
  for noise_mode in noise_modes:
    for r in noise_ratio:
      for num_sample in num_samples:
        acc = os.system("python eval_bmtl.py --r {} --leave_sub {} --noise_mode {} --num_samples {}".format(r,subid, noise_mode, num_sample))
        # acc = os.system(
        #   "python eval_no_da.py --r {} --leave_sub {} --noise_mode {}".format(r, subid, noise_mode))
