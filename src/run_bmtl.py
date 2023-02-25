# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:25:58 2022

@author: haoyu
"""

import os

noise_ratio = [0.1, 0.4]
for subid in range(1,15):
  for r in noise_ratio:
    os.system("python bmtl_trainer.py --r {} --leave_sub {}".format(r,subid))