#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:08:39 2024

@author: mcencini
"""

import deepmr

import numpy as np

import matplotlib.pyplot as plt

from deepmr.recon.calib.grog import grogop

# define object, trajectory and coils
img0 = deepmr.shepp_logan(256)
smap0 = deepmr.sensmap((32, 256, 256))
head = deepmr.radial((256), nviews=200, osf=2.0)

# nufft recon
ksp = deepmr.fft.nufft(smap0[:, None, ...] * img0, head.traj, oversamp=2.0)
img = deepmr.fft.nufft_adj(head.dcf * ksp, head.traj, head.shape, oversamp=2.0)
img = deepmr.rss(img, axis=0).squeeze()
img = abs(img)

# get sense map and calibration data
smap, cal_data = deepmr.recon.espirit_cal(ksp, head.traj, head.dcf, head.shape)

# get cartesian ksp and indexes
d, indexes, weights = grogop.grog_interp(
    ksp,
    cal_data,
    head.traj,
    head.shape,
    lamda=0.0,
)

# recon
img_grog = deepmr.fft.sparse_ifft(weights * d, indexes, head.shape)
img_grog = deepmr.rss(img_grog, axis=0).squeeze()
img_grog = abs(img_grog)

# normalize
out0 = abs(img)
out = abs(img_grog)

out0 = out0 / np.nanmax(out0)
out = out / np.nanmax(out)

plt.subplot(1, 2, 1)
plt.imshow(abs(np.concatenate((out0, out), axis=-1)), cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(abs(out0 - out), cmap="bwr"), plt.colorbar()
