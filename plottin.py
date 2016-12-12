import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default = '-1')
args = parser.parse_args()
infilename = args.filename
outfilename  = infilename[:-7]

with open(filename, 'wb') as f:
  data = pickle.load(f)

fig = plt.figure()
plt.plot(data)

