import optunity
import random
import pdb
from associative_neural_net_model import run_optimizer_on_subject
import pickle
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest='experiment', type=str, default='no-test')
    parser.add_argument('--sub', dest='sub', type=int, default=0)
    args = parser.parse_args()

    details_list = []

    pmap = optunity.parallel.create_pmap(20)

    experiment = args.experiment
    sub = args.sub

    for i in range(1):
        run = run_optimizer_on_subject(sub)
        if experiment == 'no-test':
            pars, details, _ = optunity.minimize(run, num_evals=1500, mu=[0, 0.9], sigma=[0,0.5], rho=[-0.99,0.999], solver_name='particle swarm', pmap=pmap)
        elif experiment == 'corr-test':
            pars, details, _ = optunity.minimize(run, num_evals=1500, mu=[0, 0.9], sigma=[0,0.5], rho=[-0.99,0.999], mu_t=[0, 0.9], solver_name='particle swarm', pmap=pmap)
        else:
            pars, details, _ = optunity.minimize(run, num_evals=1500, mu=[0, 0.9], sigma=[0,0.5], rho=[-0.99,0.999], mu_t=[0, 0.9], sigma_t=[0,0.5], solver_name='particle swarm', pmap=pmap)

        details_list.append(details.call_log)

        f = open('pkls/' + experiment + '_' + str(sub) + '.pkl', 'wb')
        pickle.dump(details_list, f)
