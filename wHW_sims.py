#!/usr/bin/env python
# coding: utf-8

from wLSM_utils import *
import numpy as np

ns = 500 * 10**np.arange(0, 3)
ms = ns/10
ms = ms.astype(int)

c0=5
c1=100

ase5_500, ase100_500 = wHardy_Weinberg(ns[0], ms[0], c0, c1, acorn=1)
ase5_5000, ase100_5000 = wHardy_Weinberg(ns[1], ms[1], c0, c1, acorn=1)
ase5_50000, ase100_50000 = wHardy_Weinberg(ns[2], ms[2], c0, c1, acorn=1)

import _pickle as pkl
pkl.dump(ase5_500, open('ase5_500.pkl', 'wb'))
pkl.dump(ase100_500, open('ase100_500.pkl', 'wb'))

pkl.dump(ase5_5000, open('ase5_5000.pkl', 'wb'))
pkl.dump(ase100_5000, open('ase100_5000.pkl', 'wb'))

pkl.dump(ase5_50000, open('ase5_50000.pkl', 'wb'))
pkl.dump(ase100_50000, open('ase100_50000.pkl', 'wb'))
