from tqdm import tqdm
import jax 
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def tprint(obj):
    """ helper to print training progress """
    tqdm.write(str(obj))

def nll_categorical(logits, labels):
    """ multiclass classification negative log-likelihood """

    loss = -jnp.sum(logits * labels, axis = 1) + logsumexp(logits, axis = 1)
    return jnp.mean(loss, axis = 0)

def ece(probs, y_batch, bins=20):
    """ expected calibration error, 
        source: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py """

    bin_boundaries = jnp.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = jnp.max(probs, 1)
    predictions = jnp.argmax(probs, 1)
    accuracies = (predictions == jnp.argmax(y_batch, 1))

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences >= bin_lower) * (confidences <= bin_upper)
        prob_in_bin = in_bin.astype('float32').mean()

        if prob_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].astype('float32').mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece 

def normal_like_tree(a, key):
    """ get a random gaussian variable for every parameter in tree """
    treedef = jax.tree_util.tree_structure(a)
    num_vars = len(jax.tree_util.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_map(lambda p, k: jax.random.normal(k, shape=p.shape), a,
                              jax.tree_util.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]

def auroc(probs, y_batch, bins=2500): 
    """ area under ROC as described in Appendix H, Deep Learning with Bayesian Principles """
    thresholds = jnp.linspace(1, 0, bins + 1) 
    thresholds = thresholds[1:-1] # 0.9 0.8 0.7 ... 0.1 for 10 bins

    tprs = [0.] 
    fprs = [0.]

    P = float((jnp.argmax(probs, axis=-1) == jnp.argmax(y_batch, axis=-1)).sum())  # number of positives
    N = float((jnp.argmax(probs, axis=-1) != jnp.argmax(y_batch, axis=-1)).sum())  # number of negatives

    confidences = jnp.max(probs, 1)

    for t in thresholds:
        # compute number of positives/negatives on examples with confidence bigger than t

        # select data-points above confidence threshold
        probs_above = probs[confidences >= t, :]
        ybatch_above = y_batch[confidences >= t, :]

        # True positve: confident and correct
        tp = float((jnp.argmax(probs_above, axis=-1) == jnp.argmax(ybatch_above, axis=-1)).sum())

        # False positve: confident and incorrect
        fp = float((jnp.argmax(probs_above, axis=-1) != jnp.argmax(ybatch_above, axis=-1)).sum())  

        tprs.append(tp / P) # true positive rate
        fprs.append(fp / N) # false positive rate 
            
    tprs.append(1.)
    fprs.append(1.) 

    return jnp.trapz(jnp.array(tprs), jnp.array(fprs))