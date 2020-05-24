# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['1650beam192.msh',
         '1650beam288.msh',
         '1650beam384.msh',
         '1650beam480.msh',
         '1650beam672.msh',
         '1650beam864.msh',
         '1650beam1248.msh']
with open("data_E_benchmark.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example_benchmarkt.py", beam, "--profile"], stdout=output);
    for beam in beams:
        subprocess.call(["python", "./example_benchmarkt.py", beam, "--optimised", "--profile"], stdout=output);
    for beam in beams:
        subprocess.call(["python", "./example_benchmarkt.py", beam, "--optimised", "--lumped", "--profile"], stdout=output);
    
