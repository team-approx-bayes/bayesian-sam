"""  Various neural network models in haiku 
  Inspired by: 
  https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/jax_models.py 
"""

import haiku as hk
import jax
import jax.numpy as jnp
from haiku.initializers import Constant
import functools
from resnet18 import ResNet18

he_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')
_DEFAULT_BN_CONFIG = {
    'decay_rate': 0.9,
    'eps': 1e-5,
    'create_scale': True,
    'create_offset': True
}

class FilterResponseNorm(hk.Module):
    def __init__(self, eps=1e-6, name='frn'):
        super().__init__(name=name)
        self.eps = eps

    def __call__(self, x, **unused_kwargs):
        del unused_kwargs
        par_shape = (1, 1, 1, x.shape[-1])  # [1,1,1,C]
        tau = hk.get_parameter('tau', par_shape, x.dtype, init=jnp.zeros)
        beta = hk.get_parameter('beta', par_shape, x.dtype, init=jnp.zeros)
        gamma = hk.get_parameter('gamma', par_shape, x.dtype, init=jnp.ones)
        nu2 = jnp.mean(jnp.square(x), axis=[1, 2], keepdims=True)
        x = x * jax.lax.rsqrt(nu2 + self.eps)
        y = gamma * x + beta
        z = jnp.maximum(y, tau)

        return z

def _resnet_layer(
        inputs, num_filters, normalization_layer, kernel_size=3, strides=1,
        activation=lambda x: x, use_bias=True, is_training=True
):
    x = inputs
    x = hk.Conv2D(
        num_filters, kernel_size, stride=strides, padding='same',
        w_init=he_normal, with_bias=use_bias)(x)
    x = normalization_layer()(x, is_training=is_training)
    x = activation(x)
    return x

def make_resnet_fn(
        num_classes: int,
        depth: int,
        normalization_layer,
        width: int = 16,
        use_bias: bool = True,
        activation=jax.nn.relu,
):
    num_res_blocks = (depth - 2) // 6
    if (depth - 2) % 6 != 0:
        raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

    def forward(x, is_training):
        num_filters = width
        x = _resnet_layer(
            x, num_filters=num_filters, activation=activation,
            use_bias=use_bias,
            normalization_layer=normalization_layer
        )

        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = _resnet_layer(
                    x, num_filters=num_filters, strides=strides,
                    activation=activation,
                    use_bias=use_bias, is_training=True,
                    normalization_layer=normalization_layer)
                y = _resnet_layer(
                    y, num_filters=num_filters, use_bias=use_bias,
                    is_training=True,
                    normalization_layer=normalization_layer)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match changed dims
                    x = _resnet_layer(
                        x, num_filters=num_filters, kernel_size=1,
                        strides=strides,
                        use_bias=use_bias, is_training=True,
                        normalization_layer=normalization_layer)
                x = activation(x + y)
            num_filters *= 2
        x = hk.AvgPool((8, 8, 1), 8, 'VALID')(x)
        x = hk.Flatten()(x)
        logits = hk.Linear(num_classes, w_init=he_normal)(x)
        return logits

    return forward

def make_resnet20_frn_fn(num_classes, activation=jax.nn.relu):
    return make_resnet_fn(
        num_classes, depth=20, normalization_layer=FilterResponseNorm,
        activation=activation)

def make_mlp_fn(output_dim, layer_dims, nonlinearity = jax.nn.elu):
    biasinit = Constant(0.05)
    def forward(inp, is_training):
        out = hk.Flatten()(inp)
        for layer_dim in layer_dims:
            out = hk.Linear(layer_dim, b_init=biasinit)(out)
            out = nonlinearity(out)
        return hk.Linear(output_dim, b_init=biasinit)(out)

    return forward

def make_resnet18_classification(num_classes): 
    def forward(x, is_training):
        net = ResNet18(num_classes = num_classes)
        return net(x, is_training = is_training)

    return forward

def make_lenet5_fn(num_classes):

    def lenet_fn(x, is_training):

        cnn = hk.Sequential([
            hk.Conv2D(output_channels=6, kernel_shape=5, padding="SAME"),
            jax.nn.relu,
            hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
            hk.Conv2D(output_channels=16, kernel_shape=5, padding="SAME"),
            jax.nn.relu,
            hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
            hk.Conv2D(output_channels=120, kernel_shape=5, padding="SAME"),
            jax.nn.relu,
            hk.MaxPool(window_shape=3, strides=2, padding="VALID"),
            hk.Flatten(),
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(num_classes - 1),
        ])
        return cnn(x)

    return lenet_fn

def get_model(model_name, num_classes, **kwargs):
    _MODEL_FNS = {
        "resnet20": functools.partial(
            make_resnet20_frn_fn, activation=lambda x: x),
        "mlp": functools.partial(
            make_mlp_fn, layer_dims=[500, 300], nonlinearity=jax.nn.elu),
        "resnet18": make_resnet18_classification,
        "lenet": make_lenet5_fn,
    }

    if model_name not in _MODEL_FNS.keys():
        raise NameError('Available keys:', _MODEL_FNS.keys())

    net_fn = _MODEL_FNS[model_name](num_classes, **kwargs)
    net = hk.transform_with_state(net_fn)

    return net.apply, net.init
