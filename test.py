import argparse
import os
import pickle
import sys

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from data import dataloader
from models import get_model
from util import ece, auroc, normal_like_tree

num_workers = 4

def main():
    # options and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--testbatchsize', dest='testbatchsize', 
                        type=int, default=200)
    parser.add_argument('--testmc', dest='testmc', type=int, default=1,
                        help='number of mcsamples used in bayesian model averaging')
    parser.add_argument('--resultsfolder', dest='resultsfolder', type=str, 
                        default='.', required=True)
    
    parser.set_defaults(augment = True)
    args = parser.parse_args()

    with open(os.path.join(args.resultsfolder, 'trainstate.pickle'), 'rb') as file:
        trainstate = pickle.load(file)
        trainargs = pickle.load(file)

    with open(os.path.join(args.resultsfolder, 'info.txt'), 'rt', encoding='utf-8') as file:
        info = file.read()
        print('information of the run that has been loaded:')
        print(info)

    # prepare dataset
    try:
        trainset, testset, trainloader, testloader = \
            dataloader(trainargs.dataset)(trainargs.batchsize, trainargs.testbatchsize, 
                                          trainargs.datasetfolder, trainargs.augment, num_workers)
        
    except KeyError:
        print(f'Dataset {args.dataset} not implemented.')
        sys.exit()

    ndata = len(trainset)
    ntestdata = len(testset)
    nclasses = len(trainset.classes)

    print(f"""  > dataset={trainargs.dataset} (ntrain={ndata}, """
        f"""ntest={ntestdata}, nclasses={nclasses})""")

    rngkey = jax.random.PRNGKey(trainargs.randomseed)
    np.random.seed(trainargs.randomseed)
    torch.manual_seed(trainargs.randomseed) 
    rngkey, initkey = jax.random.split(rngkey)

    # create model and optimizer
    modelapply, modelinit = get_model(trainargs.model.lower(), nclasses)

    datapoint = next(iter(trainloader))[0].numpy().transpose(0, 2, 3, 1)
    print('  > datashape (minibatch) ', datapoint.shape)
    params, netstate = modelinit(initkey, datapoint, True)
    numparams = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"""  > model='{trainargs.model}' ({numparams} parameters)""")

    rngkey, initkey = jax.random.split(rngkey)
    
    # evaluate model
    batchprobs_mean = [] 
    batchprobs_bayes = [] 
    batchlabels = [] 

    nll_mean = 0.0
    nll_bayes = 0.0 

    correct_mean = 0
    correct_bayes = 0
    total = 0

    print('testing...')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
        tgt = jax.nn.one_hot(targets.numpy(), nclasses)

        theta = trainstate.optstate['w']
        logits, _ =  modelapply(theta, trainstate.netstate, None, dat, is_training = False)
        correct_mean += jnp.sum(logits.argmax(axis=1) == tgt.argmax(axis=1))
        total += logits.shape[0]    

        nll_mean += -jnp.mean(jnp.sum(tgt * jax.nn.log_softmax(logits, axis=1), axis=1))

        batchprobs_mean.append(jax.nn.softmax(logits, axis=1))
        batchlabels.append(tgt)

        sampleprobs = [] 
        samplelogits = [] 
        for i in range(args.testmc):
            if trainargs.optim == 'bsam':
                noise, rngkey = normal_like_tree(trainstate.optstate['w'], rngkey)

                theta_sampled = jax.tree_map(lambda n, mu, s: mu + \
                    jnp.sqrt(1.0 / (ndata * trainargs.dafactor * s)) * n, noise, trainstate.optstate['w'], trainstate.optstate['s']) 
            else:
                theta_sampled = theta 
                
            logits, _ =  modelapply(theta_sampled, trainstate.netstate, None, dat, is_training = False)
            samplelogits.append(logits) 
            sampleprobs.append(jax.nn.softmax(logits, axis=1))

        bayesprobs = jnp.mean(jnp.array(sampleprobs), axis=0)
        correct_bayes += jnp.sum(bayesprobs.argmax(axis=1) == tgt.argmax(axis=1))
        batchprobs_bayes.append(bayesprobs)

        temp = jax.nn.log_softmax(jnp.array(samplelogits), axis=2) 
        nll_bayes += jnp.mean(jnp.sum(-tgt * logsumexp(temp, b=1/args.testmc, axis=0), axis=1))

    testacc_mean = 100.0 * (float(correct_mean) / float(total))
    nll_mean /= float(batch_idx) 
    ece_mean = ece(jnp.concatenate(batchprobs_mean, axis=0), 
                   jnp.concatenate(batchlabels, axis=0))
    auroc_mean = auroc(jnp.concatenate(batchprobs_mean, axis=0), 
                       jnp.concatenate(batchlabels, axis=0))

    testacc_bayes = 100.0 * (float(correct_bayes) / float(total))
    nll_bayes /= float(batch_idx) 
    ece_bayes = ece(jnp.concatenate(batchprobs_bayes, axis=0), 
                    jnp.concatenate(batchlabels, axis=0))
    auroc_bayes = auroc(jnp.concatenate(batchprobs_bayes, axis=0), 
                        jnp.concatenate(batchlabels, axis=0))
    
    print('results at mean of distribution:')
    print('  > testacc=%.2f%%, nll=%.4f, ece=%.4f, auroc=%.4f' % (testacc_mean, nll_mean, ece_mean, auroc_mean))
    print('results at model average (%d samples):' % args.testmc)
    print('  > testacc=%.2f%%, nll=%.4f, ece=%.4f, auroc=%.4f' % (testacc_bayes, nll_bayes, ece_bayes, auroc_bayes))

if __name__ == '__main__':
    main()
