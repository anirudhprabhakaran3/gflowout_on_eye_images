import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomMaskGenerator(nn.Module):
    def __init__(self, dropout_rate):
        super(RandomMaskGenerator, self).__init__()
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32).to(DEVICE)

        self.to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        return torch.bernoulli((1 - self.dropout_rate) * torch.ones(x.shape))

    def log_prob(self, x, mask):
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        dist = (1.0 - self.dropout_rate) * torch.ones(x.shape).to(DEVICE)
        probs = dist * mask + (1.0 - dist) * (1.0 - mask)
        return torch.log(probs).sum(1)


class MLP(nn.Module):
    def __init__(self, in_dim=48400, out_dim=10, hidden=None, activation=nn.LeakyReLU):
        super(MLP, self).__init__()
        if hidden is None:
            hidden = [32, 32]
        self.fc = nn.ModuleList()
        self.LN = nn.ModuleList()

        h_old = in_dim
        for h in hidden:
            self.fc.append(nn.Linear(h_old, h))
            self.LN.append(torch.nn.LayerNorm(h))
            h_old = h

        self.out_layer = nn.Linear(h_old, out_dim)
        self.activation = activation

        self.fc = self.fc.to(DEVICE)
        self.LN = self.LN.to(DEVICE)
        self.out_layer = self.out_layer.to(DEVICE)

        self.to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        for layer, ln in zip(self.fc, self.LN):
            x = self.activation()(layer(x))
            x = ln(x)
        x = self.out_layer(x)
        return x


class MLPMaskGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[32], activation=nn.LeakyReLU):
        super(MLPMaskGenerator, self).__init__()

        self.mlp = MLP(
            in_dim=in_dim, out_dim=out_dim, hidden=hidden, activation=activation
        )
        self.to(DEVICE)
        self.mlp = self.mlp.to(DEVICE)

    def _dist(self, x, T=1.0):
        x = self.mlp(x)
        x = nn.Sigmoid()(x / T)
        dist = x
        return x.to(DEVICE)

    def forward(self, x, T=1.0):
        x = x.to(DEVICE)
        probs_sampled = self._dist(x, T)
        return torch.bernoulli(probs_sampled)

    def log_prob(self, x, mask):
        dist = self._dist(x)
        probs = dist * mask + (1.0 - dist) * (1.0 - mask)
        return torch.log(probs).sum(1)

    def prob(self, x, mask):
        dist = self._dist(x, 1.0)
        probs = probs = dist * mask + (1.0 - dist) * (1.0 - mask)
        return probs


class MultiMLPMaskGenerator(nn.Module):
    def __init__(
        self, in_dim1, in_dim2, in_dim3, out_dim, hidden=[32], activation=nn.LeakyReLU
    ):
        super(MultiMLPMaskGenerator, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp1 = MLP(
            in_dim=in_dim1, out_dim=10, hidden=hidden, activation=activation
        )

        self.mlp2 = MLP(
            in_dim=in_dim2, out_dim=10, hidden=hidden, activation=activation
        )

        self.mlp3 = MLP(
            in_dim=in_dim3, out_dim=10, hidden=hidden, activation=activation
        )

        self.mlp_combine = MLP(
            in_dim=30,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation,
        )

        self.to(self.device)

    def _dist(self, x1, x2, x3, T=1.0):
        x1 = self.mlp1(x1)
        x2 = self.mlp2(x2)
        x3 = self.mlp3(x3)

        x = self.mlp_combine(torch.cat([x1, x2, x3], 1))

        x = nn.Sigmoid()(x / T)
        return x.to(self.device)

    def forward(self, x1, x2, x3, T=1.0):
        x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
        probs_sampled = self._dist(x1, x2, x3, 1.0)
        return torch.bernoulli(probs_sampled)


class CNN_(nn.Module):
    def __init__(
        self,
        image_shape=(160, 32, 32),
        out_dim=10,
        hidden=16 * 5 * 5,
        activation=nn.LeakyReLU,
    ):
        super(CNN_, self).__init__()

        n_in_channels = image_shape[0]

        self.LN = nn.LayerNorm(image_shape)
        self.conv1 = nn.Conv2d(n_in_channels, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(hidden, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, out_dim)
        self.activation = activation

        self.LN1 = nn.LayerNorm(hidden)
        self.LN2 = nn.LayerNorm(32)
        self.LN3 = nn.LayerNorm(16)

        self.to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)

        x = self.LN(x)
        x = self.pool(self.activation()(self.conv1(x)))
        x = self.pool(self.activation()(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.LN1(x)
        x = self.activation()(self.fc1(x))
        x = self.LN2(x)
        x = self.activation()(self.fc2(x))
        x = self.LN3(x)
        x = self.fc3(x)

        return x


class CNN_MLP(nn.Module):
    def __init__(self, CNN_in_dim, mlp_in_dim, out_dim=10, activation=nn.LeakyReLU):
        super(CNN_MLP, self).__init__()

        self.CNN = CNN_(image_shape=CNN_in_dim, out_dim=10, hidden=48400)
        self.MLP = MLP(in_dim=mlp_in_dim, out_dim=10)
        self.MLP_combine = MLP(in_dim=20, out_dim=out_dim)

        self.to(DEVICE)

    def forward(self, x, y):
        x, y = x.to(DEVICE), y.to(DEVICE)

        vec1 = self.CNN(x)
        vec2 = self.MLP(y)

        output = self.MLP_combine(torch.cat([vec1, vec2], 1))
        return output


def construct_conditional_mask_generators(
    n_channels, layer_dims, additional_input_dims, hiddens=None, activation=nn.ReLU
):
    mask_generators = nn.ModuleList()
    for layer_idx in range(len(layer_dims)):
        if layer_idx == 0:
            in_dim = layer_dims[0] + additional_input_dims[layer_idx]
            out_dim = n_channels[0]
        else:
            in_dim = layer_dims[layer_idx]
            for j in range(layer_idx):
                in_dim += n_channels[j]
                in_dim += additional_input_dims[j]
            out_dim = n_channels[layer_idx]
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim, out_dim=out_dim, hidden=hiddens, activation=activation
            )
        )
    return mask_generators


def construct_multiinput_conditional_mask_generators(
    n_channels, layer_dims, additional_input_dims, hiddens=None, activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    in_dim1, in_dim2, in_dim3, out_dim = 0, 0, 0, 0
    for layer_idx in range(len(layer_dims)):
        if layer_idx == 0:
            in_dim1 = n_channels[layer_idx]
            in_dim2 = layer_dims[layer_idx]
            in_dim2 = additional_input_dims[layer_idx]

            out_dim = n_channels[layer_idx]
        else:
            in_dim2 = layer_dims[layer_idx]
            in_dim3 = additional_input_dims[layer_idx]
            in_dim1 = 0

            for j in range(layer_idx):
                in_dim1 += n_channels[j]

            out_dim = n_channels[layer_idx]

        mask_generators.append(
            MultiMLPMaskGenerator(
                in_dim1,
                in_dim2,
                in_dim3,
                out_dim,
                hidden=hiddens,
                activation=activation,
            )
        )
    return mask_generators


def construct_unconditional_mask_generators(
    layer_dims, hiddens=None, activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()

    for layer_idx in range(len(layer_dims)):
        if layer_idx == 0:
            in_dim = 784
            out_dim = layer_dims[layer_idx]
        else:
            in_dim = 0
            for j in range(layer_idx):
                in_dim += layer_dims[j]
            out_dim = layer_dims[layer_idx]

        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim, out_dim=out_dim, hidden=hiddens, activation=activation
            )
        )
    return mask_generators
