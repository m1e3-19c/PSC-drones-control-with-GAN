import numpy as np
import torch

def generate_sample(batch_size, device):
    return torch.rand((batch_size, 3), device=device) * 2.0 - 1.0


def g(x, x_target, device):
    x = x.to(device)
    return torch.norm(x.mean(dim=0) - x_target.to(device))


def compute_loss_phi(
    N_omega,
    N_theta,
    batch_size,
    T,
    lambda_reg,
    variance,
    device,
    terminal_cost_fn,
    f_collision,
    G_theta_fn,
    phi_omega_fn,
):
    sigma = np.sqrt(variance)
    z = generate_sample(batch_size, device)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    x_list = [G_theta_fn(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega_fn(x, t, N_omega, terminal_cost_fn)
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val,
        (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True,
    )

    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i],
            x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True,
        )[0][:, i]
        laplacian += second_deriv

    H_phi = torch.norm(grad_phi_x, dim=-1, keepdim=True)
    loss_phi_terms = phi_omega_fn(x, torch.zeros_like(t), N_omega, terminal_cost_fn) + grad_phi_t + (sigma**2 / 2) * laplacian + H_phi
    loss_phi_mean = loss_phi_terms.mean()

    HJB_residual = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        HJB_residual[i] = torch.norm(grad_phi_t[i] + (sigma**2 / 2) * laplacian[i] + H_phi[i])
    loss_HJB = lambda_reg * HJB_residual.mean()

    return loss_phi_mean + loss_HJB + f_collision(x)


def compute_loss_G(
    N_omega,
    N_theta,
    batch_size,
    T,
    variance,
    device,
    alpha_grad_phi,
    alpha_loss_g_terms,
    alpha_target,
    alpha_formation,
    alpha_obstacle,
    alpha_collision,
    terminal_cost_fn,
    f_collision,
    f_obstacle,
    obstacles,
    f_formation,
    f_formation_old,
    initial_positions,
    nb_drones,
    f_formation_mode,
    model_name,
    G_theta_fn,
    phi_omega_fn,
    verbose=False,
):
    sigma = np.sqrt(variance)
    z = generate_sample(batch_size, device)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    x_list = [G_theta_fn(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega_fn(x, t, N_omega, terminal_cost_fn)
    phi_val.requires_grad_()
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val,
        (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True,
    )

    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i],
            x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True,
        )[0][:, i]
        laplacian += second_deriv

    H_phi = torch.norm(alpha_grad_phi * grad_phi_x, dim=-1, keepdim=True)
    loss_G_terms = grad_phi_t + (sigma**2 / 2) * laplacian + H_phi

    x_final = G_theta_fn(z, torch.ones_like(t), N_theta)
    formation_loss = 0

    nb_checkpoints = 5
    for i in range(1, nb_checkpoints + 1):
        sample_x_pushforwarded = G_theta_fn(z, torch.ones_like(t) * i / nb_checkpoints, N_theta)
        with torch.no_grad():
            initial_positions_pushforwarded = G_theta_fn(
                initial_positions.to(device),
                torch.ones(nb_drones, 1, device=device) * i / nb_checkpoints,
                N_theta,
            )

        if f_formation_mode in (1, 2):
            formation_loss += f_formation(z, sample_x_pushforwarded, initial_positions_pushforwarded)
        else:
            formation_loss += f_formation_old(sample_x_pushforwarded, device=device)

    target_loss = terminal_cost_fn(x_final)

    if verbose:
        print("-------------------------------------------------")
        print(model_name)
        print(f"{'collision_loss':20s}", f"{alpha_collision * f_collision(x).item():.3f}")
        print(f"{'obstacle_loss':20s}", f"{alpha_obstacle * f_obstacle(x, obstacles).item():.3f}")
        print(f"{'formation_loss':20s}", f"{alpha_formation * formation_loss.item():.3f}")
        print(f"{'target_loss':20s}", f"{alpha_target * target_loss.item():.3f}")
        print(f"{'H_phi':20s}", f"{H_phi.mean().item():.3f}")
        print(f"{'loss_G_terms':20s}", f"{alpha_loss_g_terms * loss_G_terms.mean().item():.3f}")

    return (
        target_loss,
        alpha_loss_g_terms * loss_G_terms.mean()
        + alpha_target * target_loss
        + alpha_formation * formation_loss
        + alpha_obstacle * f_obstacle(x, obstacles)
        + alpha_collision * f_collision(x),
    )