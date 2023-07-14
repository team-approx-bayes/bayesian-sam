import jax
import jax.numpy as jnp
import numpy as np
import copy 
from util import normal_like_tree
from typing import NamedTuple

class TrainState(NamedTuple):
    """
    collects the all the state required for neural network training
    """
    optstate: dict
    netstate: None
    rngkey: None

def build_sgd_optimizer(lossgrad,
                        learningrate : float,
                        momentum : float,
                        wdecay : float): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree_map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate

        (loss, netstate), grad = lossgrad(optstate['w'], trainstate.netstate, minibatch, is_training=True) 

        # momentum
        optstate['gm'] = jax.tree_map(
            lambda gm, g, w: momentum * gm + g + wdecay * w, optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree_map(lambda p, gm: p - learningrate * lrfactor * gm, optstate['w'], optstate['gm'])
    
        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate)

        return newtrainstate, loss

    return init, step

def build_sam_optimizer(lossgrad,
                        learningrate : float,
                        momentum : float,
                        wdecay : float,
                        rho : float,
                        msharpness : int): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree_map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)
    
    def _sam_gradient(trainstate, X_subbatch, y_subbatch):
        (_, netstate), grad = lossgrad(trainstate.optstate['w'], trainstate.netstate, (X_subbatch, y_subbatch), is_training = True) 
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(grad)]))
        perturbed_params = jax.tree_map(lambda p, g: p + rho * g / grad_norm, trainstate.optstate['w'], grad)
        (loss, netstate), perturbed_grad= lossgrad(perturbed_params, netstate, (X_subbatch, y_subbatch), is_training = True)

        return perturbed_grad, netstate, loss        

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate

        # split batch to simulate m-sharpness on one GPU
        X_batch = minibatch[0].reshape(msharpness, -1, *minibatch[0].shape[1:])
        y_batch = minibatch[1].reshape(msharpness, -1, *minibatch[1].shape[1:]) 
        grad, netstate, loss = jax.vmap(_sam_gradient, in_axes=(None, 0, 0))(trainstate, X_batch, y_batch)
        grad = jax.tree_map(lambda g : jnp.mean(g, axis=0), grad)
        netstate = jax.tree_map(lambda p : p[0], netstate) 
        loss = jnp.mean(loss)

        # momentum
        optstate['gm'] = jax.tree_map(
            lambda gm, g, w: momentum * gm + g + wdecay * w, optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree_map(lambda p, gm: p - learningrate * lrfactor * gm, optstate['w'], optstate['gm'])
    
        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate)

        return newtrainstate, loss

    return init, step


def build_bsam_optimizer(lossgrad,
                         learningrate : float,
                         beta1 : float,
                         beta2 : float,
                         wdecay : float,
                         rho : float,
                         msharpness : int,
                         Ndata : int, 
                         s_init : float, 
                         damping : float): 

    def init(weightinit, netstate, rngkey): 
        optstate = dict()
        optstate['w'] = copy.deepcopy(weightinit)
        optstate['gm'] = jax.tree_map(lambda p : jnp.zeros(shape=p.shape), weightinit)  
        optstate['alpha'] = learningrate 
        optstate['s'] = jax.tree_map(lambda p : s_init * jnp.ones(shape=p.shape), weightinit)

        return TrainState(optstate = optstate,
                          netstate = netstate,
                          rngkey = rngkey)
    
    def _bsam_gradient(trainstate, X_subbatch, y_subbatch, rngkey):
        optstate = trainstate.optstate

        # noisy sample
        noise, _ = normal_like_tree(optstate['w'], rngkey)
        noisy_param = jax.tree_map(lambda n, mu, s: mu + \
            jnp.sqrt(1.0 / (Ndata * s)) * n, noise, optstate['w'], optstate['s'])

        # gradient at noisy sample 
        (_, netstate), grad = lossgrad(noisy_param, trainstate.netstate, (X_subbatch, y_subbatch), is_training = True)

        perturbed_params = jax.tree_map(lambda p, g, s: p + rho * g / s, optstate['w'], grad, optstate['s'])
        (loss, netstate), perturbed_grad = lossgrad(perturbed_params, netstate, (X_subbatch, y_subbatch), is_training = True)

        gs = jax.tree_map(lambda g, s: jnp.sqrt(s * (g ** 2.0)), grad, optstate['s'])

        return gs, perturbed_grad, netstate, loss     

    def step(trainstate, minibatch, lrfactor):
        optstate = trainstate.optstate
        rngkey = trainstate.rngkey

        # split batch to simulate m-sharpness on one GPU
        rngkeys = jax.random.split(rngkey, msharpness + 1)
        X_batch = minibatch[0].reshape(msharpness, -1, *minibatch[0].shape[1:])
        y_batch = minibatch[1].reshape(msharpness, -1, *minibatch[1].shape[1:]) 
        gs, grad, netstate, loss = jax.vmap(_bsam_gradient, in_axes=(None, 0, 0, 0))(trainstate, X_batch, y_batch, rngkeys[0:msharpness])

        gs = jax.tree_map(lambda g : jnp.mean(g, axis=0), gs)
        grad = jax.tree_map(lambda g : jnp.mean(g, axis=0), grad)
        netstate = jax.tree_map(lambda p : p[0], netstate) 
        loss = jnp.mean(loss)

        # momentum
        optstate['gm'] = jax.tree_map(
            lambda gm, g, w: beta1 * gm + (1 - beta1) * (g + wdecay * w), optstate['gm'], grad, optstate['w'])

        # weight update 
        optstate['w'] = jax.tree_map(lambda p, gm, s: p - learningrate * lrfactor * gm / s, optstate['w'], optstate['gm'], optstate['s'])
    
        # update precision 
        optstate['s'] = jax.tree_map(lambda s, gs: beta2 * s + (1 - beta2) * (gs + damping + wdecay), 
                                     optstate['s'], gs)

        newtrainstate = trainstate._replace(
            optstate = optstate,
            netstate = netstate,
            rngkey = rngkeys[-1])

        return newtrainstate, loss

    return init, step
