#-------------------------------------------
# Author: Sneha Reddy Aenugu
# 
# Description: Hopfield network simulation
# of paired-associative recall task
#-------------------------------------------

import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import pickle
import random
from random import shuffle
import itertools
import copy
import math
import pdb 
import argparse
from multiprocessing import Pool
import time
from analyze_data import get_data_stats
import argparse


class HopNet:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((2*N, 2*N))

    def store_weights(self, A, B, mu1, mu2, pf, pb, nodes):

        N = self.N
        self.W1 = self.W[nodes[:N][:,None], nodes[:N]]
        self.W2 = self.W[nodes[:N][:,None], nodes[N:]]
        self.W3 = self.W[nodes[N:][:,None], nodes[:N]]
        self.W4 = self.W[nodes[N:][:,None], nodes[N:]]


        A_associative = (np.matrix(A).T)*np.matrix(A)
        store = np.random.rand(N,N) < mu1
        self.W1 += np.multiply(A_associative, store)

        B_associative = (np.matrix(B).T)*np.matrix(B)
        store = np.random.rand(N,N) < mu2
        self.W3 += np.multiply(B_associative, store)

        BA_associative = (np.matrix(B).T)*np.matrix(A)
        store = np.random.rand(N,N) < pf
        self.W2 += np.multiply(BA_associative, store)

        AB_associative = (np.matrix(A).T)*np.matrix(B)
        store = np.random.rand(N,N) < pb
        self.W4 += np.multiply(AB_associative, store)

        self.W[nodes[:N][:,None], nodes[:N]] = self.W1
        self.W[nodes[:N][:,None], nodes[N:]] = self.W2
        self.W[nodes[N:][:,None], nodes[:N]] = self.W3
        self.W[nodes[N:][:,None], nodes[N:]] = self.W4


    def recall(self, cue, target, nodes, reverse=True):
        N = self.N
        self.W1 = self.W[nodes[:N][:,None], nodes[:N]]
        self.W2 = self.W[nodes[:N][:,None], nodes[N:]]
        self.W3 = self.W[nodes[N:][:,None], nodes[:N]]
        self.W4 = self.W[nodes[N:][:,None], nodes[N:]]
        state = 2*np.random.randint(2, size=self.N)-1
        if reverse:
            hetero_inpt = self.W4*(np.matrix(cue).T)
        else:
            hetero_inpt = self.W2*(np.matrix(cue).T)
        for k in range(800):
            i = np.random.randint(self.N)
            if reverse:
                auto_inpt = np.dot(self.W1[i], state)
            else:
                auto_inpt = np.dot(self.W3[i], state)
            state[i] = np.sign(auto_inpt + hetero_inpt[i])
            cos_dist = np.dot(state, target)/(norm(state)*norm(target))
            if cos_dist >= 0.99:
                return 1,k
        return 0,-1


def get_pdf(mu, sigma, rho):
    means = [mu, mu]
    cov_matrix = np.array([[sigma*sigma, rho*sigma*sigma],[rho*sigma*sigma, sigma*sigma]])
    pdf = multivariate_normal(means, cov_matrix)
    return pdf


def run_simuation(pdfs, mus, sigmas, experiment):
    hopnet = HopNet(70)
    nodes = np.array(list(range(140)))
    nodes_store = np.zeros((12,140)).astype(int)

    A = 2*np.random.randint(2,size=(12,70))-1
    B = 2*np.random.randint(2,size=(12,70))-1

    
    ids = list(range(12))
    shuffle(ids)
    
    ### Store weights
    for j in ids:
        np.random.shuffle(nodes)
        nodes_store[j] = nodes
        while(1):
            pr_f, pr_b = pdfs[int(j/4)].rvs(1)
            if pr_f > 0 and pr_f < 1 and pr_b >0 and pr_b < 1:
                break
        hopnet.store_weights(A[j], B[j], mus[int(j/4)], mus[int(j/4)], pr_f, pr_b, nodes)

    
    results1 = np.zeros(12)
    reactions1 = np.zeros(12)
    shuffle(ids)
    
    ### First test
    for j in ids:
        if j%4 == 0:
            results1[j], reactions1[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        elif j%4 == 1:
            results1[j], reactions1[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        elif j%4 == 2:
            results1[j], reactions1[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)
        elif j%4 == 3:
            results1[j], reactions1[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)


        if results1[j] and experiment != 'no_test':
            if experiment == 'uncorr_test':
                while(1):
                    pr_f, pr_b = pdfs[-1].rvs(1)
                    if pr_f > 0 and pr_f < 1 and pr_b >0 and pr_b < 1:
                        break
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], pr_f, pr_b, nodes_store[j])
            else:
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], mus[-1], mus[-1], nodes_store[j])
            

    results2 = np.zeros(12)
    reactions2 = np.zeros(12)
    shuffle(ids)

    ### Second test
    for j in ids:
        if j%4 == 0:
            results2[j], reactions2[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        if j%4 == 1:
            results2[j], reactions2[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)
        if j%4 == 2:
            results2[j], reactions2[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        if j%4 == 3:
            results2[j], reactions2[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)

        if results2[j] and experiment != 'no_test':
            if experiment=='uncorr_test':
                while(1):
                    pr_f, pr_b = pdfs[-1].rvs(1)
                    if pr_f > 0 and pr_f < 1 and pr_b >0 and pr_b < 1:
                        break
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], pr_f, pr_b, nodes_store[j])
            else:
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], mus[-1], mus[-1], nodes_store[j])

    correct_first = 0
    incorrect_second_correct_first_i = 0
    incorrect_second_correct_first_r = 0
    correct_second_incorrect_first_i = 0
    correct_second_incorrect_first_r = 0

    identical_recall_diff = []
    reverse_recall_diff = []

    for i in range(12):
        result1 = results1[i]
        result2 = results2[i]

        if result1 == 1:
            correct_first += 1
        if result1 == 1 and result2 == 0:
            if i% 4 == 0 or i%4 == 3:
                incorrect_second_correct_first_i += 1
            else:
                incorrect_second_correct_first_r += 1
        if result1 == 0 and result2 == 1:
            if i% 4 == 0 or i%4 == 3:
                correct_second_incorrect_first_i += 1
            else:
                correct_second_incorrect_first_r += 1

        if result1==1 and result2 == 1:
            if i%4 == 0 or i%4 == 3:
                identical_recall_diff.append(reactions1[i]-reactions2[i])
            else:
                reverse_recall_diff.append(reactions1[i]-reactions2[i])


    stats = [correct_first, incorrect_second_correct_first_i, correct_second_incorrect_first_i, incorrect_second_correct_first_r, correct_second_incorrect_first_r]
    stats = np.array(stats).astype(float)

    return stats, [identical_recall_diff, reverse_recall_diff]


def get_g_squared_value(O, E):

    if len(np.where(E==0)[0]) > 0:
        return 1000

    O[np.where(O==0)] = 0.001

    g_squared = 0
    g_squared += 2*72*(O[0])*math.log(O[0]/E[0]) + 2*72*(1-O[0])*math.log((1-O[0])/(1-E[0]))
    g_squared += 2*72*O[0]*O[1]*math.log(O[1]/E[1]) + 2*72*O[0]*(1-O[1])*math.log((1-O[1])/(1-E[1]))
    g_squared += 2*72*(1-O[0])*O[2]*math.log(O[2]/E[2]) + 2*72*(1-O[0])*(1-O[2])*math.log((1-O[2])/(1-E[2]))
    g_squared += 2*72*O[0]*O[3]*math.log(O[3]/E[3]) + 2*72*O[0]*(1-O[3])*math.log((1-O[3])/(1-E[3]))
    g_squared += 2*72*(1-O[0])*O[4]*math.log(O[4]/E[4]) + 2*72*(1-O[0])*(1-O[4])*math.log((1-O[4])/(1-E[4]))

    return g_squared

def run_optimizer_on_subject(sub):
    def run(mu, sigma, rho, mu_t=None, sigma_t=None):
        if not mu_t and not sigma_t:
            experiment = 'no_test'
        elif mu_t and not sigma_t:
            experiment = 'corr_test'
        elif mu_t and sigma_t:
            experiment = 'uncorr_test'
        print(experiment)

        # Using the same mu and sigma for all repetitions of 1,3 and 5 
        if experiment=='no_test': 
            mus = [mu, mu, mu]
            sigmas = [sigma, sigma, sigma]
        elif experiment == 'corr_test':
            mus = [mu, mu, mu, mu_t]
            sigmas = [sigma, sigma, sigma]
        elif experiment == 'uncorr_test':
            mus = [mu, mu, mu, mu_t]
            sigmas = [sigma, sigma, sigma, sigma_t]

        nsim = 300

        data_stats = get_data_stats()

        stats_array = np.zeros(5)

        for k in range(nsim):
            if k%10 == 0:
                print(k)
            if experiment == 'corr_test':
                pdfs = [get_pdf(mus[i], sigmas[i], rho) for i in range(len(mus)-1)]
            else:
                pdfs = [get_pdf(mus[i], sigmas[i], rho) for i in range(len(mus))]

            stats, reaction_stats = run_simuation(pdfs, mus, sigmas, experiment)
            stats_array += stats

        stats_array[1] = stats_array[1]/stats_array[0]
        stats_array[2] = stats_array[2]/(12*nsim- stats_array[0])
        stats_array[3] = stats_array[3]/stats_array[0]
        stats_array[4] = stats_array[4]/(12*nsim - stats_array[0])
        stats_array[0] = stats_array[0]/(12*nsim)


        #rmsd = norm(stats_array - data_stats[int(sub)])/math.sqrt(5)
        #rmsd = norm(stats_array - np.mean(data_stats, axis=0))/math.sqrt(5)
        gsquared = get_g_squared_value(data_stats[int(sub)], stats_array)
        print(gsquared)

        return gsquared
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mus', dest='mus', type=list,  default=[0.6, 0.6, 0.6, 0.2])
    parser.add_argument('--sigmas', dest='sigmas', type=list,  default=[0.2, 0.2, 0.2, 0.2])
    parser.add_argument('--rho', dest='rho', default=0.9)
    parser.add_argument('--experiment', dest='experiment', type=str,  default='no_test')

    args = parser.parse_args()

    mu1 = 0.6; mu3 = 0.6; mu5 = 0.6; sigma1 = 0.2; sigma3 = 0.2; sigma5=0.2; rho=0.999; mu_t1 = 0.2; mu_t2 = 0.2; sigma_t = 0.01


    run = run_optimizer_on_subject(0)
    gsq = run(args.mus[0], args.sigmas[0], args.rho, args.mus[-1], args.sigmas[-1])
    print(gsq)
    



