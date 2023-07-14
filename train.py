import argparse
import os
import sys
import pickle

import numpy as np
import torch
from tqdm import trange, tqdm

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp
import haiku as hk

from data import dataloader
from models import get_model
from util import tprint, nll_categorical
from optim import build_sgd_optimizer, build_sam_optimizer, build_bsam_optimizer

num_workers = 4

def get_optimizer(args, ndata, modelapply):
    wdecay = args.priorprec / float(ndata) # weight-decay
    def nllloss(param, netstate, minibatch, is_training):
        logits, newstate = modelapply(param, netstate, None, minibatch[0], is_training)
        loss = nll_categorical(logits, minibatch[1])

        return loss, newstate

    if args.optim == 'sgd':
        optinit, optstep = build_sgd_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate = args.alpha, 
            momentum = args.beta1,
            wdecay = wdecay)
        
    elif args.optim == 'sam':
        optinit, optstep = build_sam_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate = args.alpha, 
            momentum = args.beta1,
            wdecay = wdecay,
            rho=args.rho,
            msharpness=args.batchsplit)
        
    elif args.optim == 'bsam':
        optinit, optstep = build_bsam_optimizer(
            jax.value_and_grad(nllloss, has_aux=True),
            learningrate = args.alpha, 
            beta1 = args.beta1, 
            beta2 = args.beta2,            
            wdecay = wdecay, 
            rho = args.rho, 
            msharpness = args.batchsplit,
            Ndata = ndata, 
            s_init = args.custominit,  
            damping = args.damping)
        
    else: 
        print(f'Optimizer {args.optim} not implemented.')
        sys.exit()

    return optinit, optstep

def main():
    """ training loop """

    # options and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--randomseed', dest='randomseed', type=int, default=0)
    parser.add_argument('--alpha', dest='alpha', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--rho', dest='rho', type=float, default=0.1,
                        help='parameter for SAM optimizers')
    parser.add_argument('--optim', dest='optim', type=str, default='sgd')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, 
                        help='momentum for gradient')
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999,
                        help='momentum for variance')
    parser.add_argument('--priorprec', dest='priorprec', type=float, 
                        default=25.0, help='prior precision')
    parser.add_argument('--dafactor', dest='dafactor', type=float, default=1.0,
                        help='multiplicative factor to adjust size of dataset')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=200)
    parser.add_argument('--testbatchsize', dest='testbatchsize', 
                        type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=180)
    parser.add_argument('--warmup', dest='warmup', type=int, default=5,
                        help='linear learning-rate warmup')
    parser.add_argument('--dataset', dest='dataset', 
                        type=str, default='cifar10')
    parser.add_argument('--batchsplit', dest='batchsplit', type=int, default=8,
                        help='independent perturbations on subbatches?')
    parser.add_argument('--noaugment', dest='augment', action='store_false',
                        help='no data augmentation')
    parser.add_argument('--model', dest='model', default='resnet18',
                        help='model architecture')
    parser.add_argument('--datasetfolder', dest='datasetfolder', type=str, 
                        default='datasets')
    parser.add_argument('--resultsfolder', dest='resultsfolder', type=str, 
                        default='results')
    parser.add_argument('--custominit', dest='custominit', type=float, 
                        default=1.0, help='special initialization value for variance')
    parser.add_argument('--damping', dest='damping', type=float, 
                        default=0.1, help='damping to stabilize the method')
        
    parser.set_defaults(augment = True)
    args = parser.parse_args()

    idx = 0 
    while True: 
        outpath = f"""{args.resultsfolder}/{args.dataset}_{args.model}/{args.optim}/run_{idx}"""
        if not os.path.exists(outpath):
            break 

        idx += 1 

    os.makedirs(outpath)

    print('information of this training run')
    print('\n'.join(f'  > {k}={v}' for k, v in args.__dict__.items()))
    print(f'  > results are saved in {outpath}.')

    # fix randomseeds
    rngkey = jax.random.PRNGKey(args.randomseed)
    np.random.seed(args.randomseed)
    torch.manual_seed(args.randomseed) 

    # prepare dataset 
    try:
        trainset, testset, trainloader, testloader = \
            dataloader(args.dataset)(args.batchsize, args.testbatchsize, 
                                     args.datasetfolder, args.augment, num_workers)
        
    except KeyError:
        print(f'Dataset {args.dataset} not implemented.')
        sys.exit()

    ndata = len(trainset)
    ntestdata = len(testset)
    nclasses = len(trainset.classes)

    print(f"""  > dataset={args.dataset} (ntrain={ndata}, """
        f"""ntest={ntestdata}, nclasses={nclasses})""")
    
    ndata *= args.dafactor  # heuristically increase size of data-set to account for data augmentation

    # prepare model
    modelapply, modelinit = get_model(args.model.lower(), nclasses)
    rngkey, initkey = jax.random.split(rngkey)

    datapoint = next(iter(trainloader))[0].numpy().transpose(0, 2, 3, 1)
    print('  > datashape (minibatch) ', datapoint.shape)
    
    params, netstate = modelinit(initkey, datapoint, True)
    numparams = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"""  > model='{args.model}' ({numparams} parameters)""")

    # prepare optimizer
    rngkey, initkey = jax.random.split(rngkey)
    optinit, optstep = get_optimizer(args, ndata, modelapply)

    def train_epoch(trainstate, lrfactor): 
        losses = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            X = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            y = jax.nn.one_hot(targets.numpy(), nclasses)

            trainstate, loss = optstep(trainstate, (X, y), lrfactor)
            losses.append(float(loss))

        return trainstate, jnp.mean(jnp.array(losses))

    def testacc(trainstate): 
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(testloader):
            dat = jnp.array(inputs.numpy().transpose(0, 2, 3, 1))
            tgt = jax.nn.one_hot(targets.numpy(), nclasses)

            theta = trainstate.optstate['w']
            logits, _ = modelapply(theta, trainstate.netstate, None, dat, is_training = False)

            correct += jnp.sum(logits.argmax(axis=1) == tgt.argmax(axis=1))
            total += logits.shape[0]    

        return float(correct) / float(total)

    trainstate = optinit(params, netstate, rngkey)
    optstep = jax.jit(optstep)

    # main loop
    total_time = 0.0 
    for epoch in trange(args.epochs + 1, 
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', smoothing=1.):
        # learning rate scheduler
        if epoch < args.warmup:
            lrfactor = jnp.linspace(0.0, 1.0, args.warmup + 1)[epoch + 1]
        else:
            step_t = float(epoch - args.warmup) / float(args.epochs + 1 - args.warmup)
            lrfactor = 0.5 * (1.0 + jnp.cos(jnp.pi * step_t))

        # train one epoch
        trainstate, loss = train_epoch(trainstate, lrfactor)

        # save intermediate results
        acc = testacc(trainstate) * 100.0
        tprint(f"""[{epoch:3d}/{args.epochs}] Trainloss (at samples): {loss:.3f}"""
               f""" | Acc: {acc:.3f} """)
            
        with open(os.path.join(outpath, 'trainstate.pickle'), 'wb') as file:
            pickle.dump(trainstate, file)
            pickle.dump(args, file)

        with open(os.path.join(outpath, 'info.txt'), 'wt', encoding='utf-8') as file:
            file.write('\n'.join(f'{k}={v}' for k,
                       v in args.__dict__.items()))
            file.write('\n')
            file.write(f"""[{epoch:3d}/{args.epochs}] Trainloss: {loss:.3f}"""
                       f""" | Acc: {acc:.3f} """)

if __name__ == '__main__':
    main()
