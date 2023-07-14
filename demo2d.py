import jax 
import jax.numpy as jnp
import jax.flatten_util as fu 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns; sns.set_theme()
from models import get_model
import optim
import numpy as np
from sklearn.datasets import make_moons
from jax.config import config
from PIL import Image

config.update("jax_enable_x64", True)
colorlist = [ '#8F5F3F', '#FFFFFF', '#435571' ]
mycmap = LinearSegmentedColormap.from_list('mycmap', colorlist, N=256) 
mpl.rcParams['figure.dpi'] = 150


# hyperparameters
batchsize = 30
learningrate = 0.5
beta1 = 0.9 
beta2 = 0.999
rho = 0.05
damping = 0.1 
epochs = 250
wdecay = 0.0001
seed = jax.random.PRNGKey(0)

# make dataset 
N = 100
np.random.seed(1)
X, y = make_moons(N, noise=0.1)
X_train = jnp.array(X, dtype=float)
y_train = jnp.array(y, dtype=float)

# model and loss function
net_apply, net_init = get_model('mlp', num_classes=1, layer_dims=[32, 32, 32, 32])
rngkey, net_init_key = jax.random.split(seed, 2)
params, netstate = net_init(net_init_key, X_train, True)

def loss_fn(param, state, minibatch, is_training = True):
    logits, new_state = net_apply(param, state, None, minibatch[0], is_training)
    loss = jnp.mean(-minibatch[1] * logits + jnp.log(1 + jnp.exp(logits))) 

    return (loss, new_state) 

optinit, optstep = optim.build_bsam_optimizer(
    jax.value_and_grad(loss_fn, has_aux=True),
    learningrate = learningrate, 
    beta1 = beta1, 
    beta2 = beta2,            
    wdecay = wdecay, 
    rho = rho, 
    msharpness = 5,
    Ndata = N, 
    s_init = 1.0,  
    damping = damping)

# prepare optimization step
optstep = jax.jit(optstep) 
trainstate = optinit(params, netstate, rngkey)

weights = [] 
variances = [] 

for epoch in range(epochs + 1): 
    # update learning rate
    t = float(epoch) / float(epochs)
    lrfactor = 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # shuffle data set
    rngkey, shufflekey = jax.random.split(rngkey, 2)
    X_train = jax.random.permutation(shufflekey, X_train)
    y_train = jax.random.permutation(shufflekey, y_train)

    # get number of batches
    N = X_train.shape[0]
    num_batches = int(jnp.ceil(N / batchsize))

    losses = [] 
    for batch_idx in range(num_batches):
        # get minibatch
        batch_start = batch_idx * batchsize
        batch_end = min(N, (batch_idx + 1) * batchsize)
        X_batch = X_train[batch_start:batch_end, :]
        y_batch = y_train[batch_start:batch_end].reshape(-1, 1)
            
        # skip irregular-sized batch 
        if X_batch.shape[0] != batchsize:
            continue 

        # do optimization step 
        trainstate, loss = optstep(trainstate, (X_batch, y_batch), lrfactor)
        losses.append(loss)

    fb_loss = np.mean(losses)
    weights.append(trainstate.optstate['w'])
    variances.append(trainstate.optstate['s'])
    
    if epoch % 10 == 0:
        print(epoch, fb_loss)

images = []
for j in range(0, epochs + 1, 2):
    paramvec, unflat = fu.ravel_pytree(weights[j])
    svec, unflat = fu.ravel_pytree(variances[j])

    V = 1.0 / (svec * N)

    num_samples = 20
    rngkey, key2 = jax.random.split(rngkey)
    param_samples = paramvec.reshape(-1, 1).repeat(num_samples, axis=1) + jnp.sqrt(V).reshape((len(paramvec), 1)) * jax.random.normal(key2, shape=(len(paramvec),num_samples))
    param_samples = param_samples.T 

    # plot data set
    fig = plt.figure()
    colors = np.array([[1,0,0], [0,0,1]])
    ax = fig.add_subplot(111)

    ax.set_xlim([-1.5, 2.7])
    ax.set_ylim([-1.0, 1.4])

    Nplot = 100
    xaxis = jnp.linspace(-1.5, 2.7, Nplot)
    yaxis = jnp.linspace(-1.0, 1.4, Nplot)
    xx, yy = jnp.meshgrid(xaxis, yaxis)
    Xtest = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1) 

    logits, _ = net_apply(weights[j], trainstate.netstate, None, Xtest, False)
    for i in range(num_samples):
        pred_logits, _ = net_apply(unflat(param_samples[i]), netstate, None, Xtest, False)

        probs_ran = 1.0 / (1.0 + jnp.exp(pred_logits))
        if jnp.min(probs_ran) < 0.5 and jnp.max(probs_ran) > 0.5:  
            ax.contour(xaxis, yaxis, probs_ran.reshape(Nplot, Nplot), [0.5], colors='gray', linewidths=0.5)

    probs = 1.0 / (1.0 + jnp.exp(logits))

    ax.contour(xaxis, yaxis, probs.reshape(Nplot, Nplot), [0.5], colors='black', linewidths=4)
    ax.scatter(X[y==0,0], X[y==0, 1], c=colorlist[-1], marker='o', edgecolors=colorlist[-1], s=60, linewidth=2)
    ax.scatter(X[y==1,0], X[y==1, 1], c=colorlist[0], marker='X', edgecolors=colorlist[0], s=80, linewidth=2)
    ax.set_axis_off()
    filename = 'frame.png'
    plt.savefig(filename, bbox_inches='tight')

    plt.close()

    image = Image.open(filename).convert("RGB")
    images.append(image)

# Save the list of images as a GIF
output_file = 'animation.gif'
images[0].save(output_file, save_all=True, append_images=images[1:], duration=50, loop=0)
