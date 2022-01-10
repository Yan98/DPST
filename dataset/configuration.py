#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats


MZ_MAX          = 3000 
MAX_LEN         = 50    #pep len
MAX_NUM_PEAK    = 500   #number of mz point

#####
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_H          = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_CO = 27.9949

aa2mass = {'<pad>': 0.0,
           '<sos>': mass_N_terminus-mass_H,
           '<eos>': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'n': 115.02695,
           'D': 115.02694, # 3
           'C': 160.03065, # C(+57.02)
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'q': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'm': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, #16
           'W': 186.07931, #17
           'Y': 163.06333, #18
           'V': 99.06841,  #19
           's': 166.99836, #20
           't': 181.01401, #21
           'y': 243.02966, #22
          }


aa2mass = {k:v for k,v in aa2mass.items() if len(k) != 1 or k.isupper()}
vocab_reserve = list(aa2mass.keys())[3:]
mass_ID_np = np.array(list(aa2mass.values()))

######
KNAPSACK_AA_RESOLUTION = 10000
mass_AA_min =  57.02146
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) #
distance_matrix = (np.array([list(range(int(0.1 * KNAPSACK_AA_RESOLUTION * 2 + 1)))] * len(aa2mass)) - 0.1 * KNAPSACK_AA_RESOLUTION) / KNAPSACK_AA_RESOLUTION
guass_pdf = scipy.stats.norm(0, 0.1)
distance_pdf = lambda x : guass_pdf.pdf(x)
