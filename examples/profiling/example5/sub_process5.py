# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:21:16 2020

@author: Ben Boys
"""

import subprocess

beams = ['1650beam792.msh', '1650beam2652.msh', '1650beam3570.msh', '1650beam4095.msh', '1650beam6256.msh', '1650beam15840.msh', '1650beam32370.msh', '1650beam74800.msh', '1650beam144900.msh', '1650beam247500.msh']
#beams = ['1650beam792t.msh', '1650beam2652t.msh', '1650beam3570t.msh', '1650beam4095t.msh', '1650beam6256t.msh', '1650beam15840t.msh', '1650beam32370t.msh', '1650beam74800t.msh', '1650beam144900t.msh']
#beams = ['3300beam952.msh', '3300beam2970.msh', '3300beam4392.msh', '3300beam6048.msh', '3300beam11836.msh', '3300beam17600.msh', '3300beam31680.msh', '3300beam64350.msh', '3300beam149600.msh']
#beams = ['3300beam952t.msh', '3300beam2970t.msh', '3300beam4392t.msh', '3300beam6048t.msh', '3300beam11836t.msh', '3300beam17600t.msh', '3300beam31680t.msh', '3300beam64350t.msh', '3300beam149600t.msh']

#with open("data_force_optimised.txt", "w+") as output:
#    for beam in beams:
#        subprocess.call(["python", "./example5.py", beam, "--optimised", "--profile"], stdout=output);
with open("data_displacement_optimised.txt", "w+") as output:
    for beam in beams:
        subprocess.call(["python", "./example5d.py", beam, "--optimised", "--lumped", "--profile"], stdout=output);