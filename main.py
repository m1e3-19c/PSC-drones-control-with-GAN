##################################################################################################################################################################################################################
#                                                                     DRONES CONTROL VIA GAN MODEL
#                                                       Participants:
#                                                       Mohamed-Reda Salhi: mohamed-reda.salhi@polytehnique.edu
#                                                       Joseph Combourieu: joseph.combourieu@polytechnique.edu
#                                                       Mohssin Bakraoui : mohssin.bakraoui@polytechnique.edu
#                                                       Andrea Bourelly: andrea.bourelly@polytechnique.edu
#                                                       In collaboration with MBDA
##################################################################################################################################################################################################################





##############################################################
#BLOCK 1: Importations
##############################################################
import sys
import pathlib
import math as m
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Params:
if len(sys.argv) < 17:
    print("usage : python3 main.py <[train / load]> <model_name> <total_time> <variance> <epsilon> <alpha_loss_g_terms> <alpha_target> <alpha_formation> <alpha_obstacle> <alpha_collision> <alpha_grad_phi> <fonction de cout> <nb drones> <formation voulue au départ> <formation voulue à l'arrivée> <obstacles>")
    exit(1)

TOTAL_TIME = float(sys.argv[3])
VARIANCE = float(sys.argv[4])
EPSILON = float(sys.argv[5])
ALPHA_LOSS_G_TERMS = float(sys.argv[6])
ALPHA_TARGET = float(sys.argv[7])
ALPHA_FORMATION = float(sys.argv[8])
ALPHA_OBSTACLE = float(sys.argv[9])
ALPHA_COLLISION = float(sys.argv[10])
ALPHA_GRAD_PHI = float(sys.argv[11])
F_FORMATION = int(sys.argv[12]) # entier pour la fonction de cout de formation (0 = pas de rotation autorisée, 1 = rotation autorisée avec Kabsch, 2 = rotation autorisée avec umeyama)
F_FORMATION_NAME = "kabsch" if F_FORMATION == 1 else "umeyama" if F_FORMATION == 2 else "no-rotation"
NB_DRONES = int(sys.argv[13])
CHOSEN_INITIAL_FORMATION = int(sys.argv[14]) # Entier pour la formation des drones : (0 = ligne droite, 1 = cercle, 2 = triangle plein)
CHOSEN_FINAL_FORMATION = int(sys.argv[15]) # Entier pour la formation des drones : (0 = ligne droite, 1 = cercle, 2 = triangle plein)
ENVIRONMENT = int(sys.argv[16]) # Entier pour la configuration d'obstacles voulue : (0 = rien, 1 = 1 mur avec virage à faire, 2 = mur avec trou, 3 = deux grosses boules)

if len(sys.argv) >= 19:
    MAX_EPOCHS = int(sys.argv[17])
    WRITE_RESULT_TO = sys.argv[18]
else:
    MAX_EPOCHS = None
    WRITE_RESULT_TO = None


INITIAL_BARYCENTER = torch.tensor([0, -0.5, 0], device=device)
FINAL_BARYCENTER = torch.tensor([0, 1.25, 0], device=device)

TRAIN = (sys.argv[1] in ("train", "t"))

PATH = pathlib.Path(sys.argv[0]).resolve().parent
BASE_MODEL_NAME = sys.argv[2]
MODEL_NAME = (
    f"{BASE_MODEL_NAME}_"
    f"{NB_DRONES}-drones_"
    f"T-{TOTAL_TIME}_"
    f"variance-{VARIANCE}_"
    f"eps-{EPSILON}_"
    f"alphaG-{ALPHA_LOSS_G_TERMS}_"
    f"alphaTarget-{ALPHA_TARGET}_"
    f"alphaForm-{ALPHA_FORMATION}_"
    f"alphaObst-{ALPHA_OBSTACLE}_"
    f"alphaCol-{ALPHA_COLLISION}_"
    f"alphaGradPhi-{ALPHA_GRAD_PHI}_"
    f"config-{CHOSEN_INITIAL_FORMATION}-{CHOSEN_FINAL_FORMATION}-{ENVIRONMENT}-{F_FORMATION_NAME}"
)
PATH_MODEL_N_OMEGA = PATH / "models" / (MODEL_NAME + "_N_omega")
PATH_MODEL_N_THETA = PATH / "models" / (MODEL_NAME + "_N_theta")

SIGMA = np.sqrt(VARIANCE)

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(), skip_weight=0.5):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.skip_weight = skip_weight

    def forward(self, x):
        return self.activation(self.linear(x)) + self.skip_weight * x


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock2 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock3 = ResBlock(hidden_dim, hidden_dim, activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return self.output_layer(x)


# Networks for the GAN-like mean-field game control formulation.
# NOmega approximates the value function (phi network)
class NOmega(nn.Module):
    def __init__(self):
        super(NOmega, self).__init__()
        # Input: 3 (state) + 1 (time) = 4; Output: scalar
        self.net = ResNet(input_dim=4, output_dim=1, activation=nn.Tanh())

    def forward(self, x, t):
        input_data = torch.cat([x, t], dim=-1)
        return self.net(input_data)


# NTheta approximates the generator.
class NTheta(nn.Module):
    def __init__(self):
        super(NTheta, self).__init__()
        # Input: 3 (latent) + 1 (time) = 4; Output: 3 (state)
        self.net = ResNet(input_dim=4, output_dim=3, activation=nn.Tanh())

    def forward(self, z, t):
        input_data = torch.cat([z, t], dim=-1)
        return self.net(input_data)

def phi_omega(x, t, N_omega):
    """
    Constructs the value function with boundary condition:
    φ_ω(x, t) = (1 - t) * N_omega(x, t) + t * g(x)
    """
    return (1 - t) * N_omega(x, t) + t * g(x)


def G_theta(z, t, N_theta:NTheta):
    """
    Constructs the generator with boundary condition:
    G_θ(z, t) = (1 - t) * z + t * N_theta(z, t)
    """
    return (1 - t) * z + t * N_theta(z, t)

##############################################################
#BLOCK 2: Costs
##############################################################
'''
SUB-BLOCK: Formation cost
'''
A = 0.1

# boite = [
#     [x, 1.5, z]
#     for x in torch.linspace(-0.5, 0.5, 6) for z in torch.linspace(-0.5, 0.5, 6)
# ] + [
#     [-0.5, y, z]
#     for y in torch.linspace(0.5, 1.5, 6) for z in torch.linspace(-0.5, 0.5, 6)
# ] + [
#     [0.5, y, z]
#     for y in torch.linspace(0.5, 1.5, 7) for z in torch.linspace(-0.5, 0.5, 6)
# ] + [
#     [x, y, -0.5]
#     for x in torch.linspace(-0.5, 0.5, 6) for y in torch.linspace(0.5, 1.5, 7)
# ] + [
#     [x, y, 0.5]
#     for x in torch.linspace(-0.5, 0.5, 6) for y in torch.linspace(0.5, 1.5, 7)
# ]


OBSTACLES = []
OBSTACLE_SIZE = 0.1

def set_obstacles(configuration):
    global INITIAL_BARYCENTER, FINAL_BARYCENTER, OBSTACLES, OBSTACLE_SIZE
    if configuration == 0: # Pas d'obstacle
        INITIAL_BARYCENTER = torch.tensor([0, -0.5, 0], device=device)
        FINAL_BARYCENTER = torch.tensor([0, 1.25, 0], device=device)

    elif configuration == 1: # Juste un mur à contourner
        OBSTACLES += [
            [x, 0.4, z]
            for x in torch.linspace(-0.5, 0.2, 5) for z in torch.linspace(-0.5, 0.5, 6)
        ]
        INITIAL_BARYCENTER = torch.tensor([-0.25, -0.25, 0], device=device)
        FINAL_BARYCENTER = torch.tensor([0, 1.25, 0], device=device)

    elif configuration == 2:
        OBSTACLES += [
            [x, 0.4, z]
            for x in torch.linspace(-0.5, -0.2, 3) for z in torch.linspace(-0.5, 0.5, 6)
        ] + [
            [x, 0.4, z]
            for x in torch.linspace(0.2, 0.5, 3) for z in torch.linspace(-0.5, 0.5, 6)
        ]

        if CHOSEN_INITIAL_FORMATION == 3:
            INITIAL_BARYCENTER = torch.tensor([0, 0.4, 0], device=device)
        else:
            INITIAL_BARYCENTER = torch.tensor([0, -0.5, 0], device=device)
            
        if CHOSEN_FINAL_FORMATION == 3:
            FINAL_BARYCENTER = torch.tensor([0, 0.4, 0], device=device)
        else:
            FINAL_BARYCENTER = torch.tensor([0, 1.25, 0], device=device)
    else:
        OBSTACLES += [
            [0.3, 0.2, 0],
            [-0.3, 0.8, 0]
        ]
        OBSTACLE_SIZE = 0.25

        INITIAL_BARYCENTER = torch.tensor([0, -0.5, 0], device=device)
        FINAL_BARYCENTER = torch.tensor([0, 1.5, 0], device=device)



set_obstacles(ENVIRONMENT)


def set_positions(nb_drones, configuration, barycenter) :
    res = torch.tensor([])
    if configuration == 0: # Ligne droite
        x = torch.linspace(-1/4, 1/4, nb_drones, device=device)
        y = torch.zeros(nb_drones, device=device) * (-1/2)
        z = torch.zeros(nb_drones, device=device)
        res = torch.stack([x, y, z], dim=1)
    elif configuration == 1:
        t = torch.linspace(-torch.pi, torch.pi * (1 - 2 / (NB_DRONES + 1))  , nb_drones, device=device)
        radius = 1 / 4.
        x = torch.cos(t) * radius
        y = torch.sin(t) * radius - 1 / 4.
        z = torch.zeros_like(x, device=device)
        res = torch.stack([x, y, z], dim=1)
    elif configuration == 2: # Les oies sauvages
        x = []
        y = []
        nb_layers = int((np.sqrt(8 * nb_drones + 1) - 1) / 2) + 1
        n = 0
        total_size = 1
        drone_gap = total_size / (2 * nb_layers)
        for i in range(nb_layers):
            for j in range(i + 1):
                if n < nb_drones:
                    n += 1
                    x.append((nb_layers - i + 2 * j) * drone_gap - total_size / 2)
                    y.append((nb_layers - i) * 1.41 * drone_gap - 0.5)

        x = torch.tensor(x, device=device)
        y = torch.tensor(y, device=device)
        z = torch.zeros_like(x, device=device)
        
        res = torch.stack([x, y, z], dim=1)
    else: # ligne verticale
        x = torch.zeros(nb_drones, device=device)
        y = torch.zeros(nb_drones, device=device) * (-1/2)
        z = torch.linspace(-1/4, 1/4, nb_drones, device=device)
        res = torch.stack([x, y, z], dim=1)

    return res - torch.mean(res, 0, keepdim=True) + barycenter
        
# variance = 0.003

def generate_density(x) :
    x_centered = x - x.mean(dim=0, keepdim=True)
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * SIGMA**2))
        norm_const = torch.tensor(2 * torch.pi * VARIANCE, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    return density_estimated

INITIAL_POSITIONS = set_positions(NB_DRONES, CHOSEN_INITIAL_FORMATION, INITIAL_BARYCENTER)
INITIAL_DENSITY = generate_density(INITIAL_POSITIONS)

FINAL_POSITIONS = set_positions(NB_DRONES, CHOSEN_FINAL_FORMATION, FINAL_BARYCENTER)
FINAL_DENSITY = generate_density(FINAL_POSITIONS)

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
    def density_estimated(pts):
        pts = pts.to(device)
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)
        gaussians = torch.exp(-dist2 / (2 * SIGMA**2))
        norm_const = torch.tensor(2 * torch.pi * VARIANCE, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    d = distance_L1_torch(FINAL_DENSITY, density_estimated, n_grid=50, device=device)
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


def f_formation(sample_x, sample_x_pushforwarded, initial_positions_pushforwarded):
    sample_x = sample_x.to(device)
    sample_x_pushforwarded = sample_x_pushforwarded.to(device)
    
    if F_FORMATION == 1: # use kabsch
        R = kabsch(initial_positions_pushforwarded, FINAL_POSITIONS)
        x_centered = sample_x_pushforwarded - sample_x_pushforwarded.mean(dim=0, keepdim=True)
        # x_centered = initial_positions_pushforwarded - initial_positions_pushforwarded.mean(dim=0, keepdim=True)
        x_centered = x_centered @ R.T
    else: # use umeyama
        R, c = umeyama(initial_positions_pushforwarded, FINAL_POSITIONS)
        x_centered = sample_x_pushforwarded - sample_x_pushforwarded.mean(dim=0, keepdim=True)
        # x_centered = initial_positions_pushforwarded - initial_positions_pushforwarded.mean(dim=0, keepdim=True)
        x_centered = c * x_centered @ R.T

    return f_formation_old(x_centered)

    # def density_estimated(pts):
    #     pts = pts.to(device)
    #     diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)
    #     dist2 = (diff ** 2).sum(dim=-1)
    #     gaussians = torch.exp(-dist2 / (2 * SIGMA**2))
    #     norm_const = torch.tensor(2 * torch.pi * VARIANCE, device=device)**(3/2)
    #     return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)
    # d = distance_L1_torch(FINAL_DENSITY, density_estimated, n_grid=50, device=device)
    # return d
'''
SUB-BLOCK: Collision Cost
'''
SECURITY_DISTANCE = 0.1

def f_collision(x_batch):
    x_batch = x_batch.to(device)
    diff = x_batch.unsqueeze(1) - x_batch.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    mask = ~torch.eye(dist_sq.size(0), dtype=torch.bool, device=dist_sq.device)
    dist_sq_no_diag = dist_sq.masked_select(mask).view(dist_sq.size(0), -1)
    loss_matrix = 1.0 / (dist_sq_no_diag + EPSILON)
    loss_matrix = loss_matrix[dist_sq_no_diag < SECURITY_DISTANCE] # On annule le coût au delà de la distance de sécurité

    if len(loss_matrix) == 0:
        return torch.tensor(0)
    else:
        return torch.sum(loss_matrix) / len(dist_sq_no_diag)

'''
SUB-BLOCK: Obstacle Costs
'''
# boite = [
#     [
#         -1/2 + i / 7,
#         -1 + j / 10,
#         1/2
#     ]
#     for i in range(7) for j in range(15)
# ] + [
#     [
#         -1/2 + i / 7,
#         -1 + j / 10,
#         -1/2
#     ]
#     for i in range(7) for j in range(15)
# ] + [
#     [
#         -1/2,
#         -1 + j / 10,
#         -1/2 + k / 10
#     ]
#     for j in range(15) for k in range(10)
# ] + [
#     [
#         1/2,
#         -1 + j / 10,
#         -1/2 + k / 10
#     ]
#     for j in range(15) for k in range(10)
# ]

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




##############################################################
#BLOCK 2:  optimization part
##############################################################
# this is part very theoritical I advise you to go read the report 

# def sample_from_wave_density(batch_size):
#     sigma = np.sqrt(variance)
#     k = 2 * m.pi / 0.5
#     x = torch.rand(batch_size, device=device) - 0.5
#     y = A * torch.sin(k * x)
#     z = torch.zeros_like(x, device=device)
#     points = torch.stack([x, y, z], dim=1)
#     noise = torch.randn_like(points, device=device) * sigma
#     return points + noise

def generate_sample(batch_size):
    return torch.rand((batch_size, 3), device=device)*2. - 1  # x dans [-1, 1]



def compute_loss_phi(N_omega, N_theta, batch_size, T, lambda_reg):
    """
    Computes the loss for the phi network using derivative and collision terms.
    """
    # Sample latent variables and time (uniform in [0, T])
    z = generate_sample(batch_size)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    # Generate states using the generator (applied sample-wise)
    x_list = [G_theta(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega(x, t, N_omega)
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val, (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True
    )
    # Approximate Laplacian: sum of second order derivatives for each spatial dimension
    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i], x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += second_deriv

    H_phi = torch.norm(ALPHA_GRAD_PHI * grad_phi_x, dim=-1, keepdim=True)
    loss_phi_terms = phi_omega(x, torch.zeros_like(t), N_omega) + grad_phi_t \
                     + (SIGMA**2 / 2) * laplacian + H_phi
    loss_phi_mean = loss_phi_terms.mean()

    # Regularization term penalizing deviation from the HJB residual.
    HJB_residual = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        HJB_residual[i] = torch.norm(
            # Penser à rajouter f_collision
            grad_phi_t[i] + (SIGMA**2 / 2)*laplacian[i] + H_phi[i]
        )
    loss_HJB = lambda_reg * HJB_residual.mean()

    return loss_phi_mean + loss_HJB + f_collision(x)


def g(x):
    x = x.to(device)
    return torch.norm(x.mean(dim=0) - FINAL_BARYCENTER.to(device))

def f_target(x):
    x = x.to(device)
    return torch.norm(x.mean(dim=0) - FINAL_BARYCENTER.to(device)) + f_formation_old(x)


def compute_loss_G(N_omega, N_theta, batch_size, T, verbose=False):
    """
    Computes the loss for the generator network.
    """
    # Sample latent variables and time (uniform in [0, T])
    z = generate_sample(batch_size)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    x_list = [G_theta(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega(x, t, N_omega)
    phi_val.requires_grad_()
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val, (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True
    )

    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i], x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += second_deriv
    
    H_phi = torch.norm(ALPHA_GRAD_PHI * grad_phi_x, dim=-1, keepdim=True)
    loss_G_terms = grad_phi_t + (SIGMA**2 / 2)*laplacian + H_phi

    x_final = G_theta(z, torch.ones_like(t), N_theta)
    formation_loss = 0
    
    nb_checkpoints = 5
    for i in range(1,nb_checkpoints + 1) :
        sample_x_pushforwarded = G_theta(z, torch.ones_like(t)*i/nb_checkpoints, N_theta)
        with torch.no_grad():
            initial_positions_pushforwarded = G_theta(INITIAL_POSITIONS.to(device), torch.ones(NB_DRONES, 1, device=device)*i/nb_checkpoints, N_theta)
        
        if F_FORMATION in (1, 2):
            formation_loss += f_formation(z, sample_x_pushforwarded, initial_positions_pushforwarded)
        else:
            formation_loss += f_formation_old(sample_x_pushforwarded, device=device)
    # Penser à rajouter f_collision et f_obstacle
    target_loss = f_target(x_final)
    # print("target_loss: " + str(target_loss))
    # print(formation_loss/5)

    if verbose:
        print("-------------------------------------------------")
        print("Au fait")
        print(f_formation(z, sample_x_pushforwarded, initial_positions_pushforwarded))
        print("-------------------------------------------------")
        print(MODEL_NAME)
        print(f"{'collision_loss':20s}", f"{ALPHA_COLLISION * f_collision(x).item():.3f}")
        print(f"{'obstacle_loss':20s}", f"{ALPHA_OBSTACLE * f_obstacle(x,OBSTACLES).item():.3f}")
        print(f"{'formation_loss':20s}", f"{ALPHA_FORMATION*formation_loss.item():.3f}")
        print(f"{'target_loss':20s}", f"{ALPHA_TARGET*target_loss.item():.3f}")
        print(f"{'H_phi':20s}", f"{H_phi.mean().item():.3f}")
        print(f"{'loss_G_terms':20s}", f"{ALPHA_LOSS_G_TERMS * loss_G_terms.mean().item():.3f}")
    return target_loss, ALPHA_LOSS_G_TERMS * loss_G_terms.mean() + ALPHA_TARGET*target_loss + ALPHA_FORMATION*formation_loss + ALPHA_OBSTACLE * f_obstacle(x, OBSTACLES) + ALPHA_COLLISION * f_collision(x)





def test_wave_trajectories(n, N_theta, N_omega, total_time=TOTAL_TIME, num_steps=100, visu=False):
    """
    For three drones initialized at the vertices of an equilateral triangle,
    generate and plot their trajectories over a total time period (in seconds).

    Args:
        N_theta: Trained generator network (instance of NTheta).
        total_time: Total simulation time (seconds).
        num_steps: Number of time samples along the trajectory.
    """

    # Prepare a list to hold trajectories for each drone
    trajectories = []  # Each entry: NumPy array of shape [num_steps, 3]
    grad_phi = []

    # Generate equally spaced time instants over the total time.
    times = torch.linspace(0, total_time, num_steps, device=device)

    for i in range(n):  # For each drone
        traj = []
        grad = []
        for t_phys in times:
            # Normalize time to [0, 1] for network input
            t_norm = t_phys / total_time
            t_tensor = torch.tensor([[t_norm]], device=device)
            z = INITIAL_POSITIONS[i:i+1]  # Shape: [1, 3]
            pos = G_theta(z, t_tensor, N_theta)  # Output: [1, 3]
            traj.append(pos[0])
            
            
            z.requires_grad_()
            t_tensor.requires_grad_()
            phi_val = phi_omega(z, t_tensor, N_omega)
            phi_val.requires_grad_()            
            grad_phi_x, grad_phi_t = torch.autograd.grad(
                phi_val, (z, t_tensor),
                grad_outputs=torch.ones_like(phi_val),
                create_graph=True
            )

            grad.append(grad_phi_x[0])
            
            if visu:
                t_norm = t_phys / total_time
                t_tensor = torch.tensor([[t_norm]], device=device)
                z = FINAL_POSITIONS[i:i+1]  # Shape: [1, 3]
                pos = G_theta(z, t_tensor, N_theta)  # Output: [1, 3]
                val = phi_omega(z, t_tensor, N_omega)
                traj.append(pos[0])
                grad.append(pos[0]) # MAUVAIS MAIS OK CAR MODE VISU
                break
        
        traj = torch.stack(traj)  # Shape: [num_steps, 3]
        grad = torch.stack(grad)
        # Detach before converting to NumPy
        trajectories.append(traj.cpu().detach().numpy())
        grad_phi.append(grad.cpu().detach().numpy())

    # with open(PATH / "trajectories" / ("trajectories_" + MODEL_NAME + ".txt"), "w") as file:
    #     file.write(str(trajectories))

    csv_path = PATH / "trajectories" / ("trajectories_" + MODEL_NAME + ".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["drone", "step", "x", "y", "z", "grad_phi_x", "grad_phi_y", "grad_phi_z"])
        for i, traj in enumerate(trajectories):
            grad = grad_phi[i]
            for step, (x, y, z) in enumerate(traj):
                grad_x, grad_y, grad_z = grad[step]
                writer.writerow([i, step, x, y, z, grad_x, grad_y, grad_z])

    # Plot the trajectories
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        traj = trajectories[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
        print("Position finale drône " + str(i) + ": " + str(traj[-1]))
    
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]  # résolution de la sphère
    for obs in OBSTACLES:
        cx, cy, cz = float(obs[0]), float(obs[1]), float(obs[2])
        x_s = cx + OBSTACLE_SIZE * np.cos(u) * np.sin(v)
        y_s = cy + OBSTACLE_SIZE * np.sin(u) * np.sin(v)
        z_s = cz + OBSTACLE_SIZE * np.cos(v)
        ax.plot_surface(x_s, y_s, z_s, color='k', alpha=0.6, linewidth=0)
    ax.set_title("Trajectories of " + str(n) + f" Drones Over {total_time: .1f} Seconds\n"+MODEL_NAME)
    ax.plot(FINAL_BARYCENTER.to("cpu")[0], FINAL_BARYCENTER.to("cpu")[1], FINAL_BARYCENTER.to("cpu")[2], 'o', c="yellow")
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

######################################################################




    

def main():
    # Hyperparameters (example values; adjust as needed)
    batch_size = 300
    T = TOTAL_TIME              # Normalized training horizon
    epochs = 2500    # Number of training iterations (increase for convergence)
    lambda_reg = 1.0
    n = NB_DRONES # Nombre de drones

    loss_phi_history = []
    loss_G_history = []

    learning_rate_phi = 4e-4
    learning_rate_gen = 1e-4

    # Instantiate networks and move them to device
    N_omega = NOmega().to(device)
    optimizer_phi = optim.Adam(N_omega.parameters(), lr=learning_rate_phi,
                               betas=(0.5, 0.9), weight_decay=1e-4)
    try:
        checkpoint = torch.load(PATH_MODEL_N_OMEGA, weights_only=True, map_location=torch.device(device=device))
        epoch = checkpoint["epoch"]
        N_omega.load_state_dict(checkpoint["model_state_dict"])
        optimizer_phi.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print(f"Impossible de charger le modèle N_omega : création d'un nouveau modèle")
        epoch=0
    else:
        N_omega.eval()

    N_theta = NTheta().to(device)
    optimizer_theta = optim.Adam(N_theta.parameters(), lr=learning_rate_gen,
                                 betas=(0.5, 0.9), weight_decay=1e-4)
    try:
        checkpoint = torch.load(PATH_MODEL_N_THETA, weights_only=True, map_location=torch.device(device=device))
        N_theta.load_state_dict(checkpoint["model_state_dict"])
        optimizer_theta.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print(f"Impossible de charger le modèle N_theta : création d'un nouveau modèle")
    else:
        N_theta.eval()

    

    # Training loop
    target = 2
    target = 900000
    
    visu = True
    infinite = True
    while TRAIN and (infinite or target > 0.1 or cout > 200) and (MAX_EPOCHS is None or epoch <= MAX_EPOCHS):
    # while epoch < epochs:
        optimizer_phi.zero_grad()
        loss_phi_val = compute_loss_phi(N_omega, N_theta, batch_size, T, lambda_reg)
        loss_phi_val.backward()
        optimizer_phi.step()

        optimizer_theta.zero_grad()
        target, loss_gen_val = compute_loss_G(N_omega, N_theta, batch_size, T, verbose=epoch%10 == 0)
        loss_gen_val.backward()
        cout = loss_gen_val
        optimizer_theta.step()

        # Update training history
        loss_phi_history.append(loss_phi_val.item())
        loss_G_history.append(loss_gen_val.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss_φ: {loss_phi_val.item():.4f} | Loss_G: {loss_gen_val.item():.4f} | target: {target} | cout {cout}")

        if epoch % 100 == 0:
            print("Sauvegarde des modèles...")
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": N_omega.state_dict(),
                        "optimizer_state_dict": optimizer_phi.state_dict()
                    },
                    PATH_MODEL_N_OMEGA
                )
            except:
                print(f"impossible de sauvegarder le fichier {PATH_MODEL_N_OMEGA}")
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": N_theta.state_dict(),
                        "optimizer_state_dict": optimizer_theta.state_dict()

                    }, PATH_MODEL_N_THETA)
            except:
                print(f"impossible de sauvegarder le fichier {PATH_MODEL_N_THETA}")

            if epoch % 500 == 0 and epoch > 0:
                save_loss_history(loss_phi_history, loss_G_history, f"figures/{MODEL_NAME}_loss_history_epoch_{epoch}.png")

        epoch += 1

    # After training, test by plotting trajectories of n drones over 20 seconds.
    if WRITE_RESULT_TO is not None:
        with open(WRITE_RESULT_TO, "a") as file:
            writer = csv.writer(file)
            
            writer.writerow([BASE_MODEL_NAME, TOTAL_TIME, VARIANCE, EPSILON, ALPHA_LOSS_G_TERMS, ALPHA_TARGET, ALPHA_FORMATION, ALPHA_OBSTACLE, ALPHA_COLLISION, ALPHA_GRAD_PHI, F_FORMATION, NB_DRONES, CHOSEN_INITIAL_FORMATION, CHOSEN_FINAL_FORMATION, ENVIRONMENT, f_target(G_theta(INITIAL_POSITIONS.to(device), torch.ones(NB_DRONES, 1, device=device)*TOTAL_TIME, N_theta))])
    test_wave_trajectories(n, N_theta, N_omega, total_time=TOTAL_TIME, num_steps=20, visu=visu)

    



if __name__ == "__main__":
    main()

