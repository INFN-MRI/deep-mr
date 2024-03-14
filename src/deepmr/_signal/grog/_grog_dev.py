#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:08:39 2024

@author: mcencini
"""

import deepmr

import torch
import numpy as np

import matplotlib.pyplot as plt

from deepmr._signal.grog import grogop

# define object, trajectory and coils
img0 = deepmr.shepp_logan(128)
smap0 = deepmr.sensmap((8, 128, 128))
head = deepmr.radial((128), nviews=200, osf=2.0)

# nufft recon
ksp = deepmr.fft.nufft(smap0[:, None, ...] * img0, head.traj)
img = deepmr.fft.nufft_adj(head.dcf * ksp, head.traj, head.shape)

# get sense map and calibration data
smap, cal_data = deepmr.recon.espirit_cal(ksp, head.traj, head.dcf, head.shape)

# get cartesian ksp and indexes
d, indexes, weights = grogop.grappa_interp(ksp, cal_data, head.traj, head.shape, lamda=0.05)

