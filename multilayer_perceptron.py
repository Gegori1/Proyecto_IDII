import numpy as np


class MultilayerPerceptron:
    
    def __init__(self, n_neurons:int, eta:float=1, tol=1e-4, max_epochs:int=1e3, random_state:int=None, print_error:bool=False, threshold:float=0.5):
        self.n_neurons = n_neurons
        self.eta = eta
        self.tol = tol
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.print_error = print_error
        self.threshold = threshold
        
        
    def __repr__(self) -> str:
        return (
            f"MultilayerPerceptronSimple(n_neurons={self.n_neurons}, eta={self.eta}, tol={self.tol}, "
            f"max_epochs={self.max_epochs}, random_state={self.random_state}, "
            f"print_error={self.print_error}, threshold={self.threshold})"
        )
    
    def initialize_w(self, N, M):
        """
        Inicializa los pesos de la red neuronal
        """
        
        L = self.n_neurons
        # check for type, empty list and positive values
        if not isinstance(self.n_neurons, int):
            raise TypeError("n_neurons debe ser un entero")
        
        if self.n_neurons <= 0:
            raise ValueError("n_neurons debe ser un entero positivo")
        
        # set seed if random_state is not None
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # first layer
        W0 = np.random.uniform(-1, 1, (L, N))
        # second layer
        Wh = np.random.uniform(-1, 1, (M, L))
        
        return W0, Wh
    
    def activation(self, x):
        """
        Función de activación
        """
        return 1 / (1 + np.exp(-self.eta*x))
    
    def fit(self, X, y):
        if len(y.shape) == 1 and y.shape[0] == X.shape[0]:
            y = y.reshape(y.shape[0], 1)
        # get dimensions
        Q, N, M = X.shape[0], X.shape[1], y.shape[1]
        # initialize weights
        W0, Wh = self.initialize_w(N, M)
        # update weights
        error_ = []
        E, epoch = np.inf, 0
        while E > self.tol and epoch < self.max_epochs:
            for q in range(Q):
                xi = np.reshape(X[q, :], (N, 1))
                di = np.reshape(y[q, :], (M, 1))
                # FORWARD
                neth = W0 @ xi
                yh = self.activation(neth)
                net0 = Wh @ yh
                y_ = self.activation(net0)
                # BACKWARD
                delta0 = (di.T - y_) * y_ * (1 - y_)
                deltah = yh * (1 - yh) * (Wh.T @ delta0)
                # update weights
                Wh += self.eta * delta0 @ yh.T
                W0 += self.eta * deltah @ xi.T
            # calculate error
            E = np.linalg.norm(delta0)
            if self.print_error:
                print(f"Epoch {epoch}: {E}")
            error_.append(E)
            # update epoch
            epoch += 1

        self.epoch_ = epoch
        self.error_ = error_
        self.W = W0, Wh
        return self
    
    def predict(self, X, prob=False):
        W0, Wh = self.W
        Q, N = X.shape[0], X.shape[1]
        neth = W0 @ X.T
        yh = self.activation(neth)
        net0 = Wh @ yh
        y_ = self.activation(net0)
        if not prob:
            y_ = np.where(y_ > self.threshold, 1, 0)
            
        return y_.T