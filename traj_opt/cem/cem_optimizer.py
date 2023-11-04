import torch
import torch.distributions as dist
import time
from tqdm import tqdm
class CEM_Optimizer:
    def __init__(self, cost_fcn, x_min = None, x_max = None) -> None:
        self.cost_fcn = cost_fcn
        self.x_min, self.x_max = x_min, x_max

    def cem_iter(self, x_init, num_samples = 500, num_iter = 10, elite_ratio = 0.2, sigma = 0.2):
        # x_init has size [1, horizon, nu]
        _, horizon, nu = x_init.size()
        device = x_init.device

        dim = horizon*nu

        # we have fixed mean and cov initialization
        mean = torch.zeros(dim).to(device)
        # cov = torch.eye(dim).to(device)
        cov = torch.ones(dim).to(device)

        cost_fcn = self.cost_fcn
        # initialize mean and cov
        for i in tqdm(range(num_iter), desc="CEM Iters", leave=False):
            start_time = time.time()
            # x_samples = dist.MultivariateNormal(mean, cov).sample((num_samples,))
            x_samples = dist.Normal(mean, cov).sample((num_samples,))
            if self.x_min is not None:
                x_samples = torch.clamp(x_samples, min = self.x_min)

            if self.x_max is not None:
                x_samples = torch.clamp(x_samples, max = self.x_max)

            input_samples = x_samples.view((-1, 1, horizon, nu))
            scores = cost_fcn(input_samples).view(-1)

            # minimize the cost function
            _, elite_idx = torch.topk(scores, int(num_samples * elite_ratio), largest=False)
            elite_samples = x_samples[elite_idx]
            mean = elite_samples.mean(dim=0)
            # cov = torch.diag(elite_samples.var(dim=0)) 
            cov = elite_samples.var(dim=0) + 1e-6
            run_time = time.time() - start_time
            print('cem iter {:d} takes {:.2f} secs. cost min: {:.2f}, cost max: {:.2f}'.format(i, run_time, scores.min().item(), scores.max().item()))
        
        best_x = elite_samples[0].view((1, horizon, nu))
        best_score = cost_fcn(best_x)
        return best_x, best_score 