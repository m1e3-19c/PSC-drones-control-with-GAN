import sys
import os
import csv
import random
import pathlib
import time


PATH = pathlib.Path(sys.argv[0]).resolve().parent
CSV_DIR_PATH = PATH / "random_search_results" 

if __name__ == "__main__":
    for filename in os.listdir(CSV_DIR_PATH):
        if filename[-4:] == ".csv":
            with open(CSV_DIR_PATH / filename, "r") as file:
                reader = csv.reader(file, delimiter=",")
                first_line = True
                for row in reader:
                    if first_line:
                        first_line = False
                    else:
                        NAME = row[0]
                        TOTAL_TIME = row[1]
                        VARIANCE = row[2]
                        EPSILON = row[3]
                        ALPHA_LOSS_G_TERMS = row[4]
                        ALPHA_TARGET = row[5]
                        ALPHA_FORMATION = row[6]
                        ALPHA_OBSTACLE = row[7]
                        ALPHA_COLLISION = row[8]
                        ALPHA_GRAD_PHI = row[9]
                        F_FORMATION = row[10]
                        NB_DRONES = row[11]
                        CHOSEN_INITIAL_FORMATION = row[12]
                        CHOSEN_FINAL_FORMATION = row[13]
                        ENVIRONMENT = row[14]
                        

                        os.system(f"./venv/bin/python3 main.py load {NAME} {TOTAL_TIME} {VARIANCE} {EPSILON} {ALPHA_LOSS_G_TERMS} {ALPHA_TARGET} {ALPHA_FORMATION} {ALPHA_OBSTACLE} {ALPHA_COLLISION} {ALPHA_GRAD_PHI} {F_FORMATION} {NB_DRONES} {CHOSEN_INITIAL_FORMATION} {CHOSEN_FINAL_FORMATION} {ENVIRONMENT}")

