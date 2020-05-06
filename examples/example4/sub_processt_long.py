# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['3300beam495000t.msh']
with open("data_force_3300beam495000t.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example4t.py", beam, "--profile"], stdout=output);
# =============================================================================
# with open("data_displacement_optimised_3300.txt", "w+") as output:
#     for beam in beams:
#         subprocess.call(["python", "./example4d.py", beam, "--optimised", "--profile"], stdout=output);
# =============================================================================
