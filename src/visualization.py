import matplotlib.pyplot as plt
import torch
import numpy as np

def test_wave_trajectories(
    n,
    N_theta,
    G_theta_fn,
    initial_positions,
    device,
    x_target,
    obstacles,
    obstacle_size,
    total_time=1.0,
    num_steps=100,
):
    """
    For n drones initialized at the configured initial positions,
    generate and plot their trajectories over a total time period (in seconds).

    Args:
        N_theta: Trained generator network (instance of NTheta).
        total_time: Total simulation time (seconds).
        num_steps: Number of time samples along the trajectory.
    """

    # Prepare a list to hold trajectories for each drone
    trajectories = []  # Each entry: NumPy array of shape [num_steps, 3]

    # Generate equally spaced time instants over the total time.
    times = torch.linspace(0, total_time, num_steps, device=device)

    for i in range(n):  # For each drone
        traj = []
        for t_phys in times:
            # Normalize time to [0, 1] for network input
            t_norm = t_phys / total_time
            t_tensor = torch.tensor([[t_norm]], device=device)
            z = initial_positions[i:i+1].to(device)  # Shape: [1, 3]
            pos = G_theta_fn(z, t_tensor, N_theta)  # Output: [1, 3]
            traj.append(pos[0])
        traj = torch.stack(traj)  # Shape: [num_steps, 3]
        # Detach before converting to NumPy
        trajectories.append(traj.cpu().detach().numpy())


    # Plot the trajectories
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        traj = trajectories[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
        print("Position finale drône " + str(i) + ": " + str(traj[-1]))
    
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]  # résolution de la sphère
    for obs in obstacles:
        cx, cy, cz = float(obs[0]), float(obs[1]), float(obs[2])
        x_s = cx + obstacle_size * np.cos(u) * np.sin(v)
        y_s = cy + obstacle_size * np.sin(u) * np.sin(v)
        z_s = cz + obstacle_size * np.cos(v)
        ax.plot_surface(x_s, y_s, z_s, color='k', alpha=0.6, linewidth=0)
    ax.set_title("Trajectories of " + str(n) + f" Drones Over {total_time: .1f} Seconds")
    ax.plot(x_target[0], x_target[1], x_target[2], 'o', c="yellow")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_box_aspect([1,2,1])
    ax.legend()
    plt.show()

def save_loss_history(loss_phi_history, loss_G_history, path):
    """ Saves the training loss history for both φ and G, side by side in two plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loss_phi_history, label='Loss φ')
    ax1.set_title('Training Loss for φ')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(loss_G_history, label='Loss G', color='orange')
    ax2.set_title('Training Loss for G')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
