# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['3300beam952.msh',
         '3300beam2970.msh',
         '3300beam4392.msh',
         '3300beam6048.msh',
         '3300beam11836.msh',
         '3300beam17600.msh',
         '3300beam31680.msh',
         '3300beam64350.msh',
         '3300beam149600.msh',
         '3300beam495000.msh']

with open("data_EC_benchmark.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example_benchmark.py", beam, "--profile"], stdout=output);
    for beam in beams:
        subprocess.call(["python", "./example_benchmark.py", beam, "--optimised", "--profile"], stdout=output);
    for beam in beams:
        subprocess.call(["python", "./example_benchmark.py", beam, "--optimised", "--lumped", "--profile"], stdout=output);
    for beam in beams:
        subprocess.call(["python", "./example_benchmark.py", beam, "--optimised", "--lumped2", "--profile"], stdout=output);
    
