
import math as m
import numpy as np
import torch

from .obstacles import OBSTACLE_SIZE, obstacles, boite


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_DRONES = 4
EPSILON = 1e-4
F_FORMATION = 0


##############################################################
#BLOCK 2: Costs
##############################################################
'''
SUB-BLOCK: Formation cost
'''
A = 0.1
#the generate wave function is just a way for us to make a list of positions in the form  of a wave for the generate_density function bu can be changed by just List [N,3]
#that  has the positions you wish to have
def generate_wave(n_samples) :
    # k = 2*m.pi/0.5
    # x = torch.linspace(-0.5, 0.5, n_samples, device=device)
    # y = A * torch.sin(k * x)
    # z = torch.ones_like(x, device=device) * 0
    # return torch.stack([x, y, z], dim=1)
    k = 2*m.pi/0.5
    x = torch.linspace(-1/4, 1/4, n_samples, device=device)
    y = torch.ones_like(x, device=device) * (-1/2)
    z = torch.zeros_like(x, device=device)
    return torch.stack([x, y, z], dim=1)

variance = 0.003

def generate_density(x) :
    x_centered = x - x.mean(dim=0, keepdim=True)
    sigma = np.sqrt(variance)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    return density_estimated

initial_positions = generate_wave(NB_DRONES)
density_real = generate_density(initial_positions)

def sample_from_density(density_func, n_samples, bounds=(-1, 1), M=None):
    """Méthode du rejet pour échantilloner selon la densité"""
    if M is None:
        # estimer M grossièrement
        pts = (torch.rand(10000, 3, device=device) * (bounds[1]-bounds[0]) + bounds[0])
        M = density_func(pts).max().item() * 1.2 + 1e-9
    samples = []
    while len(samples) < n_samples:
        batch = (torch.rand(n_samples, 3, device=device) * (bounds[1]-bounds[0]) + bounds[0])
        u = torch.rand(n_samples, device=device) * M
        keep = u <= density_func(batch)
        samples.extend(batch[keep].split(1))
    return torch.cat(samples[:n_samples], dim=0)

def distance_L1_torch(p_func, q_func, n_grid, a=-1.0, b=2.0, device=device):
    coords = torch.linspace(a, b, n_grid, device=device)
    dx = (b - a) / n_grid
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
    flat_grid = grid.view(-1, 3)
    p_vals = p_func(flat_grid)
    q_vals = q_func(flat_grid)
    return torch.sum(torch.abs(p_vals - q_vals)) * dx ** 3

def f_formation_old(x, device=device):
    x = x.to(device)
    x_centered = x - x.mean(dim=0, keepdim=True)
    sigma = np.sqrt(variance)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    d = distance_L1_torch(density_real, density_estimated, n_grid=50, device=device)
    return d

# def compute_covariance_matrix(centered_samples):
#     """
#     Calcule la matrice de covariance sur les données CENTRÉES.
#     """
#     # Échantillonnage uniforme :
#     # samples = torch.rand(num_samples, 3) * (limits[1] - limits[0]) + limits[0]
    
#     # barycentre selon la valeur de la densité en chaque point :
#     # densities = density_func(samples)
#     # weights = densities / torch.sum(densities)
#     # weights_2d = weights.unsqueeze(1).repeat(1, 3)
#     # m = torch.sum(samples) / len(samples)

#     # soustraction du barycentre :
#     # centered_samples = samples - m
    
#     # Matrice de covariance sur données centrées
#     cov_mat = torch.zeros((3, 3), device=device)
#     for i in range(3):
#         for j in range(3):
#             cov_mat[i,j] = torch.sum(centered_samples[:,i] * centered_samples[:,j])
    
#     return cov_mat

# Pour comparer deux formations :
# def get_sorted_eigenvalues(cov_matrix):
#     """
#     Retourne les valeurs propres d'une matrice de covariance triées par ordre croissant.
#     """
#     eigenvalues = torch.linalg.eigvals(cov_matrix).double().to(device)
#     eigenvalues.sort()
#     return eigenvalues[0]

# def eigen_distance_squared(eigenvalues1, eigenvalues2):
#     """
#     Calcule la distance entre deux listes de valeurs propres triées.
#     """
#     return torch.sum((eigenvalues1 - eigenvalues2)**2)

# def compare_densities(density_func1, density_func2):
#     cov_mat_1 = compute_covariance_matrix_centered(density_func1, limits=(-10, 10), num_samples=500000)
#     cov_mat_2 = compute_covariance_matrix_centered(density_func2, limits=(-10, 10), num_samples=500000)
#     eigenvalues_1 = get_sorted_eigenvalues(cov_mat_1)
#     eigenvalues_2 = get_sorted_eigenvalues(cov_mat_2)

#     return eigen_distance_squared(eigenvalues_1, eigenvalues_2)

# def f_formation_eigenvalues(sample_x, sample_x_pushforwarded):
#     x1 = sample_x.to(device)
#     x2 = sample_x_pushforwarded.to(device)

#     x1_centered = x1 - x1.mean(dim=0, keepdim=True)
#     x2_centered = x2 - x2.mean(dim=0, keepdim=True)

#     cov_mat_1 = compute_covariance_matrix(x1_centered)
#     cov_mat_2 = compute_covariance_matrix(x2_centered)

#     return eigen_distance_squared(get_sorted_eigenvalues(cov_mat_1), get_sorted_eigenvalues(cov_mat_2))

def kabsch(x, y): # x et y sont des tenseurs torch de taille (N, 3), x est le nuage de référence, y le nuage à aligner
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    H = x_centered.T @ y_centered
    U, S, V = torch.svd(H) # H = U @ S @ V.T
    R = V @ U.T
    if torch.det(R) < 0:
        V = V.clone()
        V[-1, :] *= -1
    R = V @ U.T
    return R.to(device) # On renvoie la matrice de rotation optimale pour passer de x à y

def umeyama(x, y): # x et y sont des tenseurs torch de taille (N, 3), x est le nuage de référence, y le nuage à aligner
    """
    Renvoie la matrice de rotation et le facteur de scaling tels que :
    c * x_c @ R2.T = y_c
    (où x_c et y_x sont les nuages x et y recentrés)
    """
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    n, d = x.size()
    H = (y_centered.T @ x_centered) / n

    U, D, Vt = torch.svd(H)
    D = torch.diag(D).to(device)
    U = U.to(device)
    Vt = Vt.to(device)

    S = torch.eye(d).to(device)
    if U.det() * Vt.det() < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt.T


    sigma_x2 = (x_centered ** 2).sum() / n
    c = (D @ S).trace() / sigma_x2

    return R.to(device), c # On renvoie la matrice de rotation optimale et le scaling pour passer de x à y

# TODO UMEYAMA
# def umeyama(x, y): # x et y sont des tenseurs torch de taille (N, 3), x est le nuage de référence, y le nuage à aligner
#     x_centered = x - x.mean(dim=0)
#     y_centered = y - y.mean(dim=0)
#     n = x.size()[1]
#     H = x_centered.T @ y_centered
#     U, D, Vt = torch.svd(H)
#     # R = Vt @ U.T
#     S = torch.eye(H.size())
#     if U.det() * Vt.det() < 0:
#         S[-1, -1] = -1

#     R = U @ S @ Vt
#     c = (D @ S).trace()
#     if torch.det(R) < 0:
#         Vt = Vt.clone()
#         Vt[-1, :] *= -1
#     R = Vt @ U.T
#     return R.to(device) # On renvoie la matrice de rotation optimale pour passer de x à y

def f_formation(sample_x, sample_x_pushforwarded, initial_positions_pushforwarded):
    sample_x = sample_x.to(device)
    sample_x_pushforwarded = sample_x_pushforwarded.to(device)
    
    if F_FORMATION == 1: # use kabsch
        R = kabsch(initial_positions_pushforwarded, initial_positions)
        x_centered = sample_x_pushforwarded - sample_x_pushforwarded.mean(dim=0, keepdim=True)
        x_centered = x_centered @ R.T
    else: # use umeyama
        R, c = umeyama(initial_positions_pushforwarded, initial_positions)
        x_centered = sample_x_pushforwarded - sample_x_pushforwarded.mean(dim=0, keepdim=True)
        x_centered = c * x_centered @ R.T

    sigma = np.sqrt(variance)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    d = distance_L1_torch(density_real, density_estimated, n_grid=50, device=device)
    return d
'''
SUB-BLOCK: Collision Cost
'''
def f_collision(x_batch):
    x_batch = x_batch.to(device)
    diff = x_batch.unsqueeze(1) - x_batch.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    mask = ~torch.eye(dist_sq.size(0), dtype=torch.bool, device=dist_sq.device)
    dist_sq_no_diag = dist_sq.masked_select(mask).view(dist_sq.size(0), -1)
    loss_matrix = 1.0 / (dist_sq_no_diag + EPSILON)
    if loss_matrix.mean() > 0.03 :
      return torch.tensor(0.0, device=device)
    else :
      return loss_matrix.mean()
    # return torch.clamp(loss_matrix.mean(), 100.0)


def f_obstacle(x, obstacles):
    eps = EPSILON
    x = x.to(device)
    cost = 0
    batch_size = x.size(0)
    for obstacle in obstacles :
        if not isinstance(obstacle, torch.Tensor):
            obstacle_tensor = torch.tensor(obstacle, device=x.device, dtype=x.dtype)
        else:
            obstacle_tensor = obstacle.to(device)
        for i in range(batch_size):
            Q = torch.norm(x[i] - obstacle_tensor)
            if Q < 2. * OBSTACLE_SIZE: # TODO : 
                cost += 1.0 / (max(Q-OBSTACLE_SIZE, 0) + eps) 
    return torch.tensor(cost / batch_size)
