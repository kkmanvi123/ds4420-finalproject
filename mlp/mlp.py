import torch
import torch.nn as nn

class BaseMLP(nn.Module):
    """Base class for MLP."""
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.0,
                 activation="relu", use_batchnorm=False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation_name = activation
        self.use_batchnorm = use_batchnorm

        self.model = self._build_network()

    def _get_activation(self):
        """Set the activation fn."""
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU
        }
        if self.activation_name not in activations:
            raise ValueError(f"Activation not configured: {self.activation_name}")
        return activations[self.activation_name]()

    def _build_network(self):
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batchnorm:
                # Apply normalizaiton on the batch
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation())
            if self.dropout > 0:
                # Apply dropout
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    # To be implemented specifically for the regression or classificaiton task
    def get_loss_fn(self):
        raise NotImplementedError

    def prepare_targets(self, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    
    
class MLPRegressor(BaseMLP):
    """Extension of base class, for regression tasks."""
    def __init__(self, input_dim, hidden_dims=None, dropout=0.0,
                 activation="relu", use_batchnorm=False):
        super().__init__(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm
        )

    # Regression uses MSE
    def get_loss_fn(self):
        return nn.MSELoss()

    def prepare_targets(self, y):
        return y.float().view(-1, 1)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        

class MLPClassifier(BaseMLP):
    """Extension of base class, for classification tasks."""
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout=0.0,
                 activation="relu", use_batchnorm=False):
        self.num_classes = num_classes
        super().__init__(
            input_dim=input_dim,
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_batchnorm
        )

    # Classificaiton uses Cross Entropy
    def get_loss_fn(self):
        return nn.CrossEntropyLoss()

    def prepare_targets(self, y):
        return y.long()

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1) # Softmax over

    def predict(self, x):
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)