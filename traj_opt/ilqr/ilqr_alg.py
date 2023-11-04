import torch 
import torch.nn as nn
import time
from tqdm import tqdm
from ilqr_mpc.mpc.mpc import QuadCost, GradMethods
from ilqr_mpc.mpc import mpc

class grounded_dynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a_lb = -0.2 
        self.a_ub = 0.2
        self.nu = 2 
        self.nx = 2

    def forward(self, s, a):
        return s + a


def ilqr_step(dyn, init_state, target_state, ilqr_iter = 5, n_batch = 1, mpc_T = 15, u_init = None):
    device = init_state.device
    
    z_init = init_state.repeat((n_batch, 1))
    nz = z_init.size(1)
    nu = dyn.nu

    goal_weights = torch.ones(nz).to(device)

    ctrl_penalty    = 10.0
    q = torch.cat((goal_weights, ctrl_penalty * torch.ones(nu).to(device)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)

    # TODO: not fully batcharized yet
    goal_state  = target_state[0]
    px          = -torch.sqrt(goal_weights) * goal_state
    p           = torch.cat((px, torch.zeros(nu).to(device)))
    p           = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)


    cost_fcn = QuadCost(Q, p)

    # tunable ilqr parameters
    linesearch_decay, max_linesearch_iter = 0.5, 10

    # TODO: enforce input constraints 
    # u_lower = dyn.a_lb*torch.ones((mpc_T, n_batch, dyn.nu)).to(device)
    # u_upper = dyn.a_ub*torch.ones((mpc_T, n_batch, dyn.nu)).to(device)

    mpc_alg = mpc.MPC(
        nz, nu, mpc_T,
        u_init=u_init,
        # u_lower= u_lower, u_upper= u_upper,
        lqr_iter=ilqr_iter,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        n_batch=n_batch,
        linesearch_decay= linesearch_decay,
        max_linesearch_iter= max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )

    ilqr_states, ilqr_inputs, ilqr_costs, record  = mpc_alg(z_init, cost_fcn, dyn)
    action = ilqr_inputs[0]
    return action, ilqr_inputs, record


def ilqr_iter(dyn, init_state, target_state, n_batch = 1, T = 10, mpc_T = 15):
    device = init_state.device
    
    z_init = init_state.repeat((n_batch, 1))
    nz = z_init.size(1)
    nu = dyn.nu

    u_init = None   

    goal_weights = torch.ones(nz).to(device)

    ctrl_penalty    = 0.0
    q = torch.cat((goal_weights, ctrl_penalty * torch.ones(nu).to(device)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(mpc_T, n_batch, 1, 1)

    # TODO: not fully batcharized yet
    goal_state  = target_state[0]
    px          = -torch.sqrt(goal_weights) * goal_state
    p           = torch.cat((px, torch.zeros(nu).to(device)))
    p           = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

    from ilqr_mpc.mpc.mpc import QuadCost, GradMethods
    from ilqr_mpc.mpc import mpc
    cost_fcn = QuadCost(Q, p)

    linesearch_decay, max_linesearch_iter = 0.2, 5

    u_lower = dyn.a_lb*torch.ones((mpc_T, n_batch, dyn.nu)).to(device)
    u_upper = dyn.a_ub*torch.ones((mpc_T, n_batch, dyn.nu)).to(device)

    ilqr_iter = 5

    mpc_alg = mpc.MPC(
        nz, nu, mpc_T,
        u_init=u_init,
        # u_lower= u_lower, u_upper= u_upper,
        lqr_iter=ilqr_iter,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        n_batch=n_batch,
        linesearch_decay= linesearch_decay,
        max_linesearch_iter= max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )

    # closed-loop simulation
    bs = 1
    lat_state_log = torch.zeros((T+1, bs, nz)).to(device)
    control_log = torch.zeros((T, bs, nu)).to(device)
    
    lat_state_log[0] = z_init
    z_t = z_init
    mpc_time = []
    for t in tqdm(range(T), desc = 'mpc'):
        start_time = time.time()

        # initialize the control inputs
        mpc_alg.u_init = u_init

        # run ilqr algorithm
        ilqr_states, ilqr_inputs, ilqr_costs, record  = mpc_alg(z_t, cost_fcn, dyn)

        action = ilqr_inputs[0]
        control_log[t] = action
        next_lat_state = dyn(z_t, action)
        z_t = next_lat_state
        lat_state_log[t+1] = next_lat_state
        
        u_init = torch.cat((ilqr_inputs[1:], torch.zeros((1, bs, nu)).to(device)), dim = 0)
        # u_init = u_init + 0.4*(torch.rand(u_init.size()).to(device) - 0.5)
        # u_init = None

        run_time = time.time() - start_time 
        mpc_time.append(run_time)
    
    grounded_states = lat_state_log
    target_grounded_state = target_state

    # compute tracking cost in the latent space 
    diff_log = lat_state_log - target_grounded_state.unsqueeze(0).repeat((T+1,1,1))
    dist_log = [diff_log[i].detach().cpu().abs().max().item() for i in range(T+1)]

    mpc_data = {'grounded_states': grounded_states, 'actions': control_log, 'mpc_time': mpc_time,
                'target_grounded_state': target_grounded_state, 
                'latent_diff_log': diff_log }

    return mpc_data
