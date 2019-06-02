#!/usr/bin/python

# Copyright 2018, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#  x, y, z binary


import sys 
from gurobipy import *

# Create a new model
m = Model("LP")
a = [3,5,4]
b = [6,7,8]
# Create variables
x = m.addVar(vtype=GRB.CONTINUOUS,name="x")
# Set objective

m.setObjective(quicksum((x - (a_l+ b_u)/2)*(x - (a_l+ b_u)/2) for a_l, b_u in zip(a,b)), GRB.MINIMIZE)
#m.setObjective(x, GRB.MINIMIZE)
# Add constraint: x + 2 y + 3 z <= 4
m.addConstrs((x <= b_u for b_u in b), "c0")
m.addConstrs((x >= a_l for a_l in a), "c1")
# Add constraint: x + y >= 1
m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % m.objVal)

