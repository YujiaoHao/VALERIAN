# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:25:58 2022
run clean source label on USCHAD with different noise patterns

@author: haoyu
"""

import os

noise_ratio = [0.1, 0.2, 0.4, 0.6]
for subid in range(1,15):
  for r in noise_ratio:
    os.system("python clean_source_labels.py --r {} --leave_sub {}".format(r,subid))