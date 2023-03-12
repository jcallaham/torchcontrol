import torch
import matplotlib.pyplot as plt

def plot_traj(traj, t_span):
    fig, axs = plt.subplots(1, 2, figsize=(12,4))
    for i in range(traj.shape[1]):
        axs[0].plot(t_span.cpu(), traj[:,i,0].detach().cpu(), 'tab:red', alpha=.3)
        axs[1].plot(t_span.cpu(), traj[:,i,1].detach().cpu(), 'tab:blue', alpha=.3)
    axs[0].set_xlabel(r'Time [s]'); axs[1].set_xlabel(r'Time [s]')
    axs[0].set_ylabel(r'p'); axs[1].set_ylabel(r'q')
    axs[0].set_title(r'Positions'); axs[1].set_title(r'Momenta')


# Plot learned vector field and trajectories in phase space
def plot_phase_space(model, traj, n_grid=50, graph_lim=torch.pi, device=torch.device('cpu')):
    fig, ax = plt.subplots(1, 1, figsize=(6,6))

    x = torch.linspace(-graph_lim, graph_lim, n_grid).to(device)
    Q, P = torch.meshgrid(x, x) ; z = torch.cat([Q.reshape(-1, 1), P.reshape(-1, 1)], 1)
    f = model.dynamics(0, z).detach().cpu()
    Fq, Fp = f[:,0].reshape(n_grid, n_grid), f[:,1].reshape(n_grid, n_grid)
    val = model.u(0, z).detach().cpu()
    # U = val.reshape(n_grid, n_grid)
    ax.streamplot(Q.T.detach().cpu().numpy(), P.T.detach().cpu().numpy(),
                    Fq.T.detach().cpu().numpy(), Fp.T.detach().cpu().numpy(), color='black', density=0.6, linewidth=0.5)

    ax.set_xlim([-graph_lim, graph_lim]) ; ax.set_ylim([-graph_lim, graph_lim])
    traj = traj.detach().cpu()
    for j in range(traj.shape[1]):
        ax.plot(traj[:,j,0], traj[:,j,1], color='tab:purple', alpha=.4)
    ax.set_title('Phase Space')
    ax.set_xlabel(r'p')
    ax.set_ylabel(r'q')