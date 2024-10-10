#!/bin/python3
# Written by G. Le Treut on 2023-09-05


#----------------------------------------------------------------------------
# imports
#----------------------------------------------------------------------------
from pathlib import Path
import logging

import numpy as np

#----------------------------------------------------------------------------
# parameters
#----------------------------------------------------------------------------
t_eval = [0., 1., 10., 100., 1000.] # time to evaluate
i_init = 0                          # initial state

#----------------------------------------------------------------------------
# function
#----------------------------------------------------------------------------
def rates_to_W_matrix(rates):
  '''
  Construct a W-matrix from the matrix of transition rates
  '''
  W = np.copy(rates)
  np.fill_diagonal(W, np.diagonal(W) - np.einsum('ba->a', W))
  return W

#----------------------------------------------------------------------------
# main script
#----------------------------------------------------------------------------
if __name__ == "__main__":
  logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

  logging.info("Computing probabilities from initial state {:d} for following times:\n".format(i_init) + ', '.join(['{:g}'.format(t) for t in t_eval]))
  # raise ValueError
  rdir = Path('.')

  # rate matrix parameters
  fpath = rdir / 'rates.dat'
  rates = np.loadtxt(fpath)

  # compute the W-matrix from the rates
  W = rates_to_W_matrix(rates)

  # diagonalize the W-matrix
  L, U = np.linalg.eig(-W)

  # re-order by ascending eigenvalue
  idx = np.argsort(np.real(L))
  L = L[idx]
  U = U[:, idx]

  # smallest eigenvalue must be zero (otherwise not a W-matrix)
  logging.info("setting smallest eigenvalue ({:g}) to 0.".format(L[0]))
  L[0] = 0.

  # invert the matrix of change of basis
  Uinv = np.linalg.inv(U)

  # define the evolution operator
  func_evol = lambda t: np.real(np.einsum('a,aj,a->j', U[i_init], Uinv, np.exp(-t*L)))

  # compute probability vectors
  P = np.array([func_evol(t) for t in t_eval])

  # put the times in the first column and write result
  P = np.concatenate([np.array(t_eval).reshape(-1,1), P], axis=1)

  fpath = 'markov_chain_probas.txt'
  np.savetxt(fname=fpath, X=P)
  logging.info("wrote probabilities to file {:s}".format(str(fpath)))


