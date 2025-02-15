import argparse
import os
import numpy as np
import random

def sigma_lambda(args):
    
    sigma=1.0
    lambda2=0.0
    if args.prior_type=='exist':
        lambda1=0.05
        add_lambda1=args.proportion*(args.confidence-0.5)/2
        lambda1+=add_lambda1
    if args.method=='nonlinear' and args.sem_type=='mim':
        if args.n_nodes==20:
            lambda1=0.1/(args.ER*args.size)
            lambda2=0.1/(args.ER*args.size)
        if args.n_nodes==40:
            lambda1=0.05/(args.ER*args.size)
            lambda2=0.05/(args.ER*args.size)
        if args.n_nodes==60:
            lambda1=0.045/(args.ER*args.size)
            lambda2=0.045/(args.ER*args.size)
    elif args.method=='nonlinear' and args.sem_type=='mlp':
        lambda1=0.02
        lambda2=0.02
    return sigma,lambda1,lambda2


def generate_prior(args,dag_true):
    np.random.seed(2024)
    w_prior = np.zeros((args.n_nodes,args.n_nodes))
    true_edges = list(np.argwhere(dag_true!=0))    
    true_edges = np.random.permutation(true_edges)
    absence_edges = list(np.argwhere(dag_true == 0))
    absence_edges = np.random.permutation(absence_edges)
    
    edge_existence=[]
    if args.prior_type == 'exist':
        edge_existence = true_edges[:int(len(true_edges)*args.proportion)].tolist()
        error_prior=absence_edges[:int(len(edge_existence)*args.error_prior_proportion)].tolist()
        for edge in edge_existence:
            w_prior[edge[0],edge[1]] = 1
        for edge in error_prior:
            w_prior[edge[0],edge[1]] = 1
    elif args.prior_type == 'forbidden':
        edge_forbidden = absence_edges[:int(len(absence_edges)*args.proportion)].tolist()
        error_prior=true_edges[:int(len(edge_forbidden)*args.error_prior_proportion)].tolist()
        for edge in edge_forbidden:
            w_prior[edge[0],edge[1]] = -1
        for edge in error_prior:
            w_prior[edge[0],edge[1]] = -1

    return w_prior,edge_existence,error_prior
