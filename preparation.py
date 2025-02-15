import argparse
import os
import numpy as np
import random

def get_config(set_arg=None):
    parser = argparse.ArgumentParser(description='experiments on various ')
    # help Display description information for candidate parameters
    parser.add_argument('--n_nodes', type=int, default=20, help='')
    parser.add_argument('--ER', type=int, default=2, help='')
    parser.add_argument('--size', type=int, default=2, help='')
    parser.add_argument('--graph_type', type=str, default='ER', help='')
    parser.add_argument('--random', type=int, default=0, help='')
    
    parser.add_argument('--method', type=str, default='linear', help='')
    parser.add_argument('--sem_type', type=str, default='gauss', help='')
    
    parser.add_argument('--prior_type', type=str, default='exist', help='')
    parser.add_argument('--confidence', type=float, default=0.9, help='')
    parser.add_argument('--proportion', type=float, default=0.9, help='')
    parser.add_argument('--error_prior_proportion', type=float, default=0., help='')
    parser.add_argument('--error_prior_type', type=str, default='reverse direct', help='')

    parser.add_argument('--alg', type=str, default='notears', help='')
    parser.add_argument('--adaptive_degree', type=float, default=1, help='')

    parser.add_argument('--test', type=int, default=1, help='')
    args = parser.parse_args()

    if set_arg != None and args.test: 
        for key in set_arg:
            setattr(args, key, set_arg[key])
    
    args.n_edges=args.ER*args.n_nodes
    return args

def get_config_real(set_arg=None):
    parser = argparse.ArgumentParser(description='experiments on various ')
    # help Display description information for candidate parameters
    
    parser.add_argument('--dataset', type=str, default='sachs', help='')

    parser.add_argument('--prior_type', type=str, default='exist', help='')
    parser.add_argument('--confidence', type=float, default=0.9, help='')
    parser.add_argument('--proportion', type=float, default=0.9, help='')
    parser.add_argument('--error_prior_proportion', type=float, default=0., help='')
    parser.add_argument('--error_prior_type', type=str, default='reverse direct', help='')

    parser.add_argument('--alg', type=str, default='notears', help='')

    parser.add_argument('--test', type=int, default=1, help='')
    args = parser.parse_args()

    if set_arg != None and args.test: 
        for key in set_arg:
            setattr(args, key, set_arg[key])
    
    return args

