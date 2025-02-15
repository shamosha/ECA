from castle.algorithms.gradient.notears.torch.nonlinear import *
import torch

class myECA_mlp_prior(NotearsNonlinear):
    def __init__(self, lambda1: float = 0.01,
                 lambda2: float = 0.01,
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,
                 hidden_layers: tuple = (10, 1),
                 expansions: int = 10,
                 bias: bool = True,
                 model_type: str = "mlp",
                 device_type: str = "cpu",
                 device_ids=None):

        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.hidden_layers = hidden_layers
        self.expansions = expansions
        self.bias = bias
        self.model_type = model_type
        self.device_type = device_type
        self.device_ids = device_ids
        self.rho, self.alpha, self.h = 1.0, 0.0, np.inf

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device
        
        self.w_prior=None
        self.prob_prior=0
    
        
    def dual_ascent_step(self, model, X):
        """
        Perform one step of dual ascent in augmented Lagrangian.

        Parameters
        ----------
        model: nn.Module
            network model
        X: torch.tenser
            sample data

        Returns
        -------
        :tuple
            cycle control parameter
        """
        def prior_loss(output, w_prior=None, prob_prior=0):
            if w_prior is None or torch.all(w_prior == 0):
                return 0
            adj_binary = torch.abs(2*torch.sigmoid(output) - 1)
            prob_exist = (adj_binary*prob_prior + (1-adj_binary)*(1-prob_prior))
            prob_forb = ((1-adj_binary)*prob_prior + adj_binary*(1-prob_prior))
            prior = torch.sum(torch.log(prob_exist[w_prior == 1])) + \
                    torch.sum(torch.log(prob_forb[w_prior == -1]))
            prior = - prior
            return prior
        
        h_new = None
        optimizer = LBFGSBScipy(model.parameters())
        X_torch = torch.from_numpy(X)
        while self.rho < self.rho_max:
            X_torch = X_torch.to(self.device)

            def closure():
                optimizer.zero_grad()
                X_hat = model(X_torch)
                loss = squared_loss(X_hat, X_torch)
                h_val = model.h_func()
                penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
                l2_reg = 0.5 * self.lambda2 * model.l2_reg()
                l1_reg = self.lambda1 * model.fc1_l1_reg()

                d = model.dims[0]
                fc1_weight = model.fc1_pos.weight - model.fc1_neg.weight  # [j * m1, i]
                fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
                A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
                #W = torch.sqrt(A+1e-9)  # [i, j]
                prior = prior_loss(A, self.w_prior, self.prob_prior) * 0.3
                
                primal_obj = loss + penalty + l2_reg + l1_reg + prior
                primal_obj.backward()
                return primal_obj

            optimizer.step(closure, self.device)  # NOTE: updates model in-place
            with torch.no_grad():
                model = model.to(self.device)
                h_new = model.h_func().item()
            if h_new > 0.25 * self.h:
                self.rho *= 10
            else:
                break
        self.alpha += self.rho * h_new
        self.h = h_new

    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=torch.tensor(w_prior,device=self.device)
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.penalty_lambda=lambda1