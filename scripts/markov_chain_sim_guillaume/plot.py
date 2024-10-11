#!/bin/python3
# Written by G. Le Treut on 2023-09-05


#----------------------------------------------------------------------------
# imports
#----------------------------------------------------------------------------
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# main script
#----------------------------------------------------------------------------
if __name__ == "__main__":
  logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
  rdir = Path('.')


  # load and plot
  fpath = 'markov_chain_probas.txt'
  data = np.loadtxt(fpath)
  times = data[:,0]
  probas = data[:, 1:]

  fig = plt.figure()
  ax = fig.gca()

  for t,P in zip(times, probas):
    ax.plot(P, '-', lw=0.5, label='t = {:g}'.format(t))

  ax.set_xlabel('state i', fontsize='medium')
  ax.set_ylabel('p_i', fontsize='medium')
  ax.set_xlim(0., probas.shape[1]-1)
  ax.set_ylim(0., None)
  ax.legend(loc='best', fontsize='medium')

  fpath = rdir / 'plot.png'
  fig.savefig(fpath, dpi=300)
  logging.info("wrote plot to file {:s}".format(str(fpath)))
  fig.clf()
  plt.close('all')
