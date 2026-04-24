LISTE = {
    "TOTAL_TIME": [0.8, 0.9, 1, 1, 1, 1, 1, 1.1, 1.2],
    "VARIANCE": [0.001, 0.003, 0.01],
    "EPSILON": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    "EXP": [0.5, 1, 2],
    "ALPHA_LOSS_G_TERMS": [0.1, 0.4, 1, 4, 10, 40, 100],
    "ALPHA_TARGET": [100, 300, 700, 1000, 3000, 7000],
    "ALPHA_FORMATION": [100, 300, 700, 1000, 3000, 7000],
    "ALPHA_OBSTACLE": [0.1, 0.4, 1, 4, 10, 40, 100, 400, 1000],
    "ALPHA_COLLISION": [0.1, 0.4, 1, 4, 10, 40, 100, 400],
    "ALPHA_GRAD_PHI": [0.7, 0.8, 0.9, 1, 1, 1, 1, 1, 1.2],
}

res = 1
for key in LISTE:
    v = len(LISTE[key])
    print(v)
    res *= v

print("total =", res)
