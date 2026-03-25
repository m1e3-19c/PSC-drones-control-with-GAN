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
import torch
import torch.optim as optim


import src.costs as costs_module
from src.networks import NOmega, NTheta, G_theta, phi_omega
from src.losses import compute_loss_phi as compute_loss_phi
from src.losses import compute_loss_G as compute_loss_G
from src.losses import g as terminal_cost_g
from src.visualization import test_wave_trajectories, save_loss_history
from src.costs import variance, initial_positions, f_collision, f_obstacle, f_formation, f_formation_old
from src.obstacles import OBSTACLE_SIZE, obstacles, boite, mur_a_passer

### Hyperparameters

# Simulation Params:

TOTAL_TIME = 1.
EPSILON = 1e-4
ALPHA_LOSS_G_TERMS = 500.
ALPHA_TARGET = 70.
ALPHA_FORMATION = 1.
ALPHA_OBSTACLE = 1.
ALPHA_COLLISION = 1.
ALPHA_GRAD_PHI = 1.
F_FORMATION = 1.

NB_DRONES = 4

# Training Params:

batch_size = 256
T = TOTAL_TIME   # Normalized training horizon
epochs = 1000000    # Number of training iterations (increase for convergence)
lambda_reg = 1.0
n = NB_DRONES # Number of drones for trajectory visualization

learning_rate_phi = 4e-4
learning_rate_gen = 1e-4



# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_target = torch.tensor([0, 1, 0], device="cpu")


if 3 < len(sys.argv) < 12:
    print("usage : python3 main.py <[train / load]> <model_name> <total_time> <epsilon> <alpha_loss_g_terms> <alpha_target> <alpha_formation> <alpha_obstacle> <alpha_collision> <alpha_grad_phi> <0 pour interdire les rotations, 1 pour kabsch, 2 pour umeyama>")
    exit(1)

if len(sys.argv) == 12:
    TOTAL_TIME = float(sys.argv[3])
    EPSILON = float(sys.argv[4])
    ALPHA_LOSS_G_TERMS = float(sys.argv[5])
    ALPHA_TARGET = float(sys.argv[6])
    ALPHA_FORMATION = float(sys.argv[7])
    ALPHA_OBSTACLE = float(sys.argv[8])
    ALPHA_COLLISION = float(sys.argv[9])
    ALPHA_GRAD_PHI = float(sys.argv[10])
    F_FORMATION = int(sys.argv[11])

costs_module.EPSILON = EPSILON
costs_module.F_FORMATION = F_FORMATION

TRAIN = (sys.argv[1] in ("train", "t"))

PATH = pathlib.Path(sys.argv[0]).resolve().parent
BASE_MODEL_NAME = sys.argv[2]
MODEL_NAME = (
    f"{BASE_MODEL_NAME}_"
    f"T-{TOTAL_TIME}_"
    f"eps-{EPSILON}_"
    f"alphaG-{ALPHA_LOSS_G_TERMS}_"
    f"alphaTarget-{ALPHA_TARGET}_"
    f"alphaForm-{ALPHA_FORMATION}_"
    f"alphaObst-{ALPHA_OBSTACLE}_"
    f"alphaCol-{ALPHA_COLLISION}_"
    f"alphaGradPhi-{ALPHA_GRAD_PHI}"
)
PATH_MODEL_N_OMEGA = PATH / "models" / (MODEL_NAME + "_N_omega")
PATH_MODEL_N_THETA = PATH / "models" / (MODEL_NAME + "_N_theta")


def main():
    # Hyperparameters (example values; adjust as needed)

    loss_phi_history = []
    loss_G_history = []

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
    
    visu = False
    infinite = True
    while TRAIN and (infinite or target > 0.1 or cout > 200):
    # while epoch < epochs:
        optimizer_phi.zero_grad()
        loss_phi_val = compute_loss_phi(
            N_omega,
            N_theta,
            batch_size,
            T,
            lambda_reg,
            variance,
            device,
            lambda x: terminal_cost_g(x, x_target, device),
            f_collision,
            G_theta,
            phi_omega,
        )
        loss_phi_val.backward()
        optimizer_phi.step()

        optimizer_theta.zero_grad()
        target, loss_gen_val = compute_loss_G(
            N_omega,
            N_theta,
            batch_size,
            T,
            variance,
            device,
            ALPHA_GRAD_PHI,
            ALPHA_LOSS_G_TERMS,
            ALPHA_TARGET,
            ALPHA_FORMATION,
            ALPHA_OBSTACLE,
            ALPHA_COLLISION,
            lambda x: terminal_cost_g(x, x_target, device),
            f_collision,
            f_obstacle,
            obstacles,
            f_formation,
            f_formation_old,
            initial_positions,
            NB_DRONES,
            F_FORMATION,
            MODEL_NAME,
            G_theta,
            phi_omega,
            verbose=epoch % 10 == 0,
        )
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

        if visu:
            break

    # After training, test by plotting trajectories of n drones over 20 seconds.
    test_wave_trajectories(
        n,
        N_theta,
        G_theta,
        initial_positions,
        device,
        x_target,
        mur_a_passer,
        OBSTACLE_SIZE,
        total_time=TOTAL_TIME,
        num_steps=20,
    )


if __name__ == "__main__":
    main()

