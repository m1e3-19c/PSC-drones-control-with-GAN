import sys
import os
import csv
import random
import pathlib
import time

LISTE = {
    "TOTAL_TIME": [0.8, 0.9, 1, 1, 1, 1, 1, 1.1, 1.2],
    "VARIANCE": [0.001, 0.003, 0.01],
    "EPSILON": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    "ALPHA_LOSS_G_TERMS": [0.1, 0.4, 1, 4, 10, 40, 100],
    "ALPHA_TARGET": [100, 300, 700, 1000, 3000, 7000],
    "ALPHA_FORMATION": [100, 300, 700, 1000, 3000, 7000],
    "ALPHA_OBSTACLE": [0.1, 0.4, 1, 4, 10, 40, 100, 400, 1000],
    "ALPHA_COLLISION": [0.1, 0.4, 1, 4, 10, 40, 100, 400],
    "ALPHA_GRAD_PHI": [0.7, 0.8, 0.9, 1, 1, 1, 1, 1, 1.2],
}


if len(sys.argv) < 7:
    print("usage : python3 random_search.py <name> <f_formation> <nb_drones> <initial_formation> <final_formation> <obstacles>")
    exit(1)

NAME = sys.argv[1]
F_FORMATION = int(sys.argv[2]) # entier pour la fonction de cout de formation (0 = pas de rotation autorisée, 1 = rotation autorisée avec Kabsch, 2 = rotation autorisée avec umeyama)
NB_DRONES = int(sys.argv[3])
CHOSEN_INITIAL_FORMATION = int(sys.argv[4]) # Entier pour la formation des drones : (0 = ligne droite, 1 = cercle, 2 = triangle plein)
CHOSEN_FINAL_FORMATION = int(sys.argv[5]) # Entier pour la formation des drones : (0 = ligne droite, 1 = cercle, 2 = triangle plein)
ENVIRONMENT = int(sys.argv[6]) # Entier pour la configuration d'obstacles voulue : (0 = rien, 1 = 1 mur avec virage à faire, 2 = mur avec trou, 3 = deux grosses boules)

if len(sys.argv) >= 8:
    if sys.argv[7] in ("no_formation", "no_f"):
        LISTE["ALPHA_FORMATION"] = [0]

TOTAL_NAME = (
    f"{NAME}_"
    f"f_form-{F_FORMATION}_"
    f"{NB_DRONES}-drones_"
    f"initial-{CHOSEN_INITIAL_FORMATION}_"
    f"final-{CHOSEN_FINAL_FORMATION}_"
    f"obst-{ENVIRONMENT}"
)

MAX_EPOCH = 2000

PATH = pathlib.Path(sys.argv[0]).resolve().parent
CSV_PATH = PATH / "random_search_results" / ("result_" + TOTAL_NAME + ".csv")

if __name__ == "__main__":
    with open(CSV_PATH, "a") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "total_time", "variance", "epsilon", "alpha_loss_g_terms", "alpha_target", "alpha_formation", "alpha_obstacle", "alpha_collision", "alpha_grad_phi", "fonction_cout", "nb_drones", "initial_formaiton", "final_formation", "obstacles", "target_loss"])

    counter = 0
    while True:
        counter += 1
        TOTAL_TIME = random.choice(LISTE["TOTAL_TIME"])
        VARIANCE = random.choice(LISTE["VARIANCE"])
        EPSILON = random.choice(LISTE["EPSILON"])
        ALPHA_LOSS_G_TERMS = random.choice(LISTE["ALPHA_LOSS_G_TERMS"])
        ALPHA_TARGET = random.choice(LISTE["ALPHA_TARGET"])
        ALPHA_FORMATION = random.choice(LISTE["ALPHA_FORMATION"])
        ALPHA_OBSTACLE = random.choice(LISTE["ALPHA_OBSTACLE"])
        ALPHA_COLLISION = random.choice(LISTE["ALPHA_COLLISION"])
        ALPHA_GRAD_PHI = random.choice(LISTE["ALPHA_GRAD_PHI"])

        os.system(f"./venv/bin/python3 main.py train {NAME + "_" + str(counter)} {TOTAL_TIME} {VARIANCE} {EPSILON} {ALPHA_LOSS_G_TERMS} {ALPHA_TARGET} {ALPHA_FORMATION} {ALPHA_OBSTACLE} {ALPHA_COLLISION} {ALPHA_GRAD_PHI} {F_FORMATION} {NB_DRONES} {CHOSEN_INITIAL_FORMATION} {CHOSEN_FINAL_FORMATION} {ENVIRONMENT} {MAX_EPOCH} {CSV_PATH}")

        # print(NAME)
        print()
        print("---------------------------------------")
        print()
        print("FIN D'ENTRAINEMENT")
        print()
        print("---------------------------------------")
        print()
        time.sleep(10)