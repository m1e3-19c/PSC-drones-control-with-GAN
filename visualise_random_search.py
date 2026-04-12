import sys
import os
import csv
import random
import pathlib
import time
import subprocess


PATH = pathlib.Path(sys.argv[0]).resolve().parent
CSV_DIR_PATH = PATH / "random_search_results" 

CONTENT = ""
if len(sys.argv) >= 2:
    CONTENT = sys.argv[1]

if __name__ == "__main__":
    for filename in os.listdir(CSV_DIR_PATH):
        if CONTENT in filename and filename[-4:] == ".csv":
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
                        EXP = row[4]
                        ALPHA_LOSS_G_TERMS = row[5]
                        ALPHA_TARGET = row[6]
                        ALPHA_FORMATION = row[7]
                        ALPHA_OBSTACLE = row[8]
                        ALPHA_COLLISION = row[9]
                        ALPHA_GRAD_PHI = row[10]
                        F_FORMATION = row[11]
                        NB_DRONES = row[12]
                        CHOSEN_INITIAL_FORMATION = row[13]
                        CHOSEN_FINAL_FORMATION = row[14]
                        ENVIRONMENT = row[15]
                        
                        subprocess_args = [
                            "./venv/bin/python3",
                            "main.py",
                            "load",
                            str(NAME),
                            str(TOTAL_TIME),
                            str(VARIANCE),
                            str(EPSILON),
                            str(EXP),
                            str(ALPHA_LOSS_G_TERMS),
                            str(ALPHA_TARGET),
                            str(ALPHA_FORMATION),
                            str(ALPHA_OBSTACLE),
                            str(ALPHA_COLLISION),
                            str(ALPHA_GRAD_PHI),
                            str(F_FORMATION),
                            str(NB_DRONES),
                            str(CHOSEN_INITIAL_FORMATION),
                            str(CHOSEN_FINAL_FORMATION),
                            str(ENVIRONMENT),
                        ]

                        if CONTENT in NAME:
                            print(" ".join(subprocess_args))
                            subprocess.run(
                                subprocess_args
                            )

