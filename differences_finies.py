import numpy as np
import scipy.linalg as la
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAÈTRES GLOBAUX
# ==========================================
N = 75                      # Résolution spatiale (21x21x21)
L = 4.0                     # Taille du domaine [-4, 4]
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)
dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

T_horizon = 3.0
Nt = 200                    # Pas de temps
dt = T_horizon / Nt

# Paramètres physiques et algorithmiques
sigma = 0.5                 # Bruit brownien
kappa = 0.5                 # Poids de la congestion (réduit pour la stabilité)
gamma_form = 3.0            # Poids de la formation
V_MAX = 5.0                 # Vitesse max (Gradient Clipping)
relax_base = 0.08           # Taux de relaxation initial

# ==========================================
# 2. POSITIONS ET DENITÉ INITIALE
# ==========================================
n_drones = 5
start_center = np.array([0, -2.5, 0])

# Positions initiales exactes des drones
np.random.seed(42) # Pour la reproductibilité
# POSITIONS_INITIALES = np.random.normal(loc=start_center, scale=0.4, size=(n_drones, 3))
POSITIONS_INITIALES = np.array(
    [
        np.linspace(-L / 3, L / 3, n_drones),
        np.linspace(0, 0, n_drones),
        np.linspace(0, 0, n_drones)
    ]
).T + start_center

# Construction de rho_0 par somme de Gaussiennes
rho_0 = np.zeros_like(X)
sigma_drone = 1.5 * dx      # Largeur de la gaussienne liée à la grille

for pos in POSITIONS_INITIALES:
    px, py, pz = pos
    drone_density = np.exp(-((X - px)**2 + (Y - py)**2 + (Z - pz)**2) / (2 * sigma_drone**2))
    rho_0 += drone_density

rho_0 /= np.sum(rho_0 * dx**3) # Normalisation stricte

# ==========================================
# 3. FONCTIONS DE COÛT (Cible, Obtacle, Formation)
# ==========================================
# Cible : Coût terminal
target_center = np.array([0, 2.5, 0])
g_terminal = 0.5 * ((X - target_center[0])**2 + (Y - target_center[1])**2 + (Z - target_center[2])**2)


OBSTACLES = [
    [x, 0.4, z]
    for x in np.linspace(-L/1.5, -L / 3.5, 3) for z in np.linspace(-L/1.75, L/1.75, 6)
] + [
    [x, 0.4, z]
    for x in np.linspace(L / 3.5, L/1.5, 3) for z in np.linspace(-L/1.75, L/1.75, 6)
]

OBSTACLE_RADIUS = 0.5

def cost_obstacle(X, Y, Z, radius, weight):
    # On initialise un tableau de zéros de la même taille que la grille 3D
    res = np.zeros_like(X) 
    
    for cx, cy, cz in OBSTACLES:
        dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
        # On ajoute une cloche gaussienne pour chaque obstacle
        # Plus radius est grand, plus l'obstacle est "large"
        res += np.exp(-dist_sq / (2 * (radius / 1.5)**2))
        
    return weight * res

F_OBS = cost_obstacle(X, Y, Z, radius=OBSTACLE_RADIUS, weight=15.0)

# Outils pour la Formation avec Rotation (PCA)
def get_principal_axes(rho, X, Y, Z, dx):
    mass = np.sum(rho) * (dx**3)
    if mass < 1e-8:
        return np.array([0,0,0]), np.eye(3)
        
    cx = np.sum(rho * X) * (dx**3) / mass
    cy = np.sum(rho * Y) * (dx**3) / mass
    cz = np.sum(rho * Z) * (dx**3) / mass
    
    cov = np.zeros((3, 3))
    dX_c, dY_c, dZ_c = X - cx, Y - cy, Z - cz
    
    cov[0, 0] = np.sum(rho * dX_c**2); cov[0, 1] = np.sum(rho * dX_c * dY_c); cov[0, 2] = np.sum(rho * dX_c * dZ_c)
    cov[1, 1] = np.sum(rho * dY_c**2); cov[1, 2] = np.sum(rho * dY_c * dZ_c)
    cov[2, 2] = np.sum(rho * dZ_c**2)
    cov[1, 0] = cov[0, 1]; cov[2, 0] = cov[0, 2]; cov[2, 1] = cov[1, 2]
    cov /= mass
    
    evals, evecs = la.eigh(cov) 
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    if la.det(evecs) < 0: evecs[:, 2] = -evecs[:, 2]
        
    return np.array([cx, cy, cz]), evecs

def evaluate_target_shape(X, Y, Z):
    # Forme cible : un essaim aplati sur l'axe Z
    return np.exp(- (X**2 / 2.0 + Y**2 / 2.0 + Z**2 / 0.1))

def cost_formation_rotative(rho_current, X, Y, Z, dx, gamma):
    center_curr, evecs_curr = get_principal_axes(rho_current, X, Y, Z, dx)
    R = evecs_curr @ np.eye(3).T # On aligne sur le repère standard
    
    dX_c, dY_c, dZ_c = X - center_curr[0], Y - center_curr[1], Z - center_curr[2]
    
    X_rot = R[0,0]*dX_c + R[1,0]*dY_c + R[2,0]*dZ_c
    Y_rot = R[0,1]*dX_c + R[1,1]*dY_c + R[2,1]*dZ_c
    Z_rot = R[0,2]*dX_c + R[1,2]*dY_c + R[2,2]*dZ_c
    
    rho_target_aligned = evaluate_target_shape(X_rot, Y_rot, Z_rot)
    mass_target = np.sum(rho_target_aligned) * (dx**3)
    if mass_target > 0:
        rho_target_aligned *= (np.sum(rho_current)*(dx**3) / mass_target)
        
    return gamma * (rho_current - rho_target_aligned)**2

# ==========================================
# 4. OPÉRATEURS DIFFÉRENTIELLES 3D
# ==========================================
def laplacian_3d(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1, 1:-1] = (f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1] + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1] + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2] - 6 * f[1:-1, 1:-1, 1:-1]) / (dx**2)
    return lap

def grad_hjb_3d(u, dx):
    gx_f, gx_b = np.zeros_like(u), np.zeros_like(u)
    gy_f, gy_b = np.zeros_like(u), np.zeros_like(u)
    gz_f, gz_b = np.zeros_like(u), np.zeros_like(u)

    gx_f[:-1, :, :] = (u[1:, :, :] - u[:-1, :, :]) / dx; gx_b[1:, :, :] = (u[1:, :, :] - u[:-1, :, :]) / dx
    gy_f[:, :-1, :] = (u[:, 1:, :] - u[:, :-1, :]) / dx; gy_b[:, 1:, :] = (u[:, 1:, :] - u[:, :-1, :]) / dx
    gz_f[:, :, :-1] = (u[:, :, 1:] - u[:, :, :-1]) / dx; gz_b[:, :, 1:] = (u[:, :, 1:] - u[:, :, :-1]) / dx

    gx = np.where(gx_f + gx_b > 0, gx_b, gx_f)
    gy = np.where(gy_f + gy_b > 0, gy_b, gy_f)
    gz = np.where(gz_f + gz_b > 0, gz_b, gz_f)
    return gx, gy, gz

def div_fp_3d(rho, vx, vy, vz, dx):
    fx, fy, fz = rho * vx, rho * vy, rho * vz
    div = np.zeros_like(rho)
    
    div[1:-1, :, :] += np.where(vx[1:-1, :, :] > 0, (fx[1:-1, :, :] - fx[:-2, :, :]) / dx, (fx[2:, :, :] - fx[1:-1, :, :]) / dx)
    div[:, 1:-1, :] += np.where(vy[:, 1:-1, :] > 0, (fy[:, 1:-1, :] - fy[:, :-2, :]) / dx, (fy[:, 2:, :] - fy[:, 1:-1, :]) / dx)
    div[:, :, 1:-1] += np.where(vz[:, :, 1:-1] > 0, (fz[:, :, 1:-1] - fz[:, :, :-2]) / dx, (fz[:, :, 2:] - fz[:, :, 1:-1]) / dx)
    return div

# ==========================================
# 5. BOUCLE PRINCIPALE MFG (Point Fixe)
# ==========================================
rho_seq = np.tile(rho_0, (Nt, 1, 1, 1))
u_seq = np.zeros((Nt, N, N, N))
max_iters = 300

print("--- Démarrage de la résolution MFG ---")
for iteration in range(max_iters):
    
    # A. HJB (Rétrograde)
    u_seq[-1] = g_terminal
    for t in range(Nt-2, -1, -1):
        lap_u = laplacian_3d(u_seq[t+1], dx)
        gx, gy, gz = grad_hjb_3d(u_seq[t+1], dx)
        
        gx, gy, gz = np.clip(gx, -V_MAX, V_MAX), np.clip(gy, -V_MAX, V_MAX), np.clip(gz, -V_MAX, V_MAX)
        norm_grad_sq = gx**2 + gy**2 + gz**2
        
        f_congestion = kappa * rho_seq[t+1]
        f_form = cost_formation_rotative(rho_seq[t+1], X, Y, Z, dx, gamma_form)
        f_total = f_congestion + F_OBS + f_form
        
        u_seq[t] = u_seq[t+1] - dt * (-0.5 * sigma**2 * lap_u + 0.5 * norm_grad_sq - f_total)
        
    # B. Extraction des Vitesses
    vx_seq, vy_seq, vz_seq = np.zeros_like(u_seq), np.zeros_like(u_seq), np.zeros_like(u_seq)
    for t in range(Nt):
        gx, gy, gz = grad_hjb_3d(u_seq[t], dx)
        vx_seq[t] = np.clip(-gx, -V_MAX, V_MAX)
        vy_seq[t] = np.clip(-gy, -V_MAX, V_MAX)
        vz_seq[t] = np.clip(-gz, -V_MAX, V_MAX)

    # C. Fokker-Planck (Progressive)
    new_rho_seq = np.zeros_like(rho_seq)
    new_rho_seq[0] = rho_0
    for t in range(0, Nt-1):
        lap_rho = laplacian_3d(new_rho_seq[t], dx)
        div_rho_v = div_fp_3d(new_rho_seq[t], vx_seq[t], vy_seq[t], vz_seq[t], dx)
        
        new_rho_seq[t+1] = new_rho_seq[t] + dt * (0.5 * sigma**2 * lap_rho - div_rho_v)
        new_rho_seq[t+1] = np.maximum(new_rho_seq[t+1], 0)
        mass = np.sum(new_rho_seq[t+1] * dx**3)
        if mass > 0: new_rho_seq[t+1] /= mass

    # D. Relaxation dynamique
    error = np.max(np.abs(new_rho_seq - rho_seq))
    current_relax = relax_base / (1.0 + 0.1 * iteration)
    rho_seq = current_relax * new_rho_seq + (1 - current_relax) * rho_seq
    
    print(f"Itération {iteration+1}/{max_iters} | Erreur max : {error:.6f} | Relax : {current_relax:.3f}")

# ==========================================
# 6. SIMULATION DES TRAJECTOIRES
# ==========================================
print("\n--- Simulation des trajectoires 3D ---")
trajectories = np.zeros((Nt, n_drones, 3))
trajectories[0] = np.copy(POSITIONS_INITIALES)
current_positions = np.copy(POSITIONS_INITIALES)

for t in range(Nt - 1):
    interp_vx = RegularGridInterpolator((x, y, z), vx_seq[t], bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((x, y, z), vy_seq[t], bounds_error=False, fill_value=0.0)
    interp_vz = RegularGridInterpolator((x, y, z), vz_seq[t], bounds_error=False, fill_value=0.0)
    
    v_current = np.column_stack((interp_vx(current_positions), interp_vy(current_positions), interp_vz(current_positions)))
    dW = np.random.normal(0, np.sqrt(dt), size=(n_drones, 3))
    
    current_positions = current_positions + v_current * dt + sigma * dW
    trajectories[t+1] = current_positions

# ==========================================
# 7. VISUALISATION 3D
# ==========================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Trajectoires
for i in range(n_drones):
    ax.scatter(trajectories[:, i, 0], trajectories[:, i, 1], trajectories[:, i, 2], alpha=1, linewidth=1.2)

# Point de départ, Cible et Obstacle
ax.scatter(start_center[0], start_center[1], start_center[2], color='blue', s=100, label='Départ Essaim', marker='s')
ax.scatter(target_center[0], target_center[1], target_center[2], color='green', s=150, label='Cible (g)', marker='*')

for obs_x, obs_y, obs_z in OBSTACLES:
    # Dessin approximatif de l'obstacle sphérique
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
    obs_x = obs_x + 0.5 * np.cos(u_sphere) * np.sin(v_sphere)
    obs_y = obs_y + 0.5 * np.sin(u_sphere) * np.sin(v_sphere)
    obs_z = obs_z + 0.5 * np.cos(v_sphere)
    ax.plot_surface(obs_x, obs_y, obs_z, color="black", alpha=0.6, linewidth=0)

ax.set_xlim([-L, L]); ax.set_ylim([-L, L]); ax.set_zlim([-L, L])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title("Navigation d'un essaim par HJB avec Évitement d'Obstacle et Formation")
ax.legend()
plt.show()