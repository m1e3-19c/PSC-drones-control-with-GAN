"""  
This module defines the obstacles.
"""

'''
SUB-BLOCK: Obstacle Costs
'''
#the list obstacles is the list you can modify to put obstacles however you like here below lies just an example of obstacles that can be configurated

import torch

OBSTACLE_SIZE = 0.1


mur_a_passer = [
    [x, 0.4, z]
    for x in torch.linspace(-0.5, -0.2, 3) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [x, 0.4, z]
    for x in torch.linspace(0.2, 0.5, 3) for z in torch.linspace(-0.5, 0.5, 6)
]

boite = [
    [x, 1.5, z]
    for x in torch.linspace(-0.5, 0.5, 6) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [-0.5, y, z]
    for y in torch.linspace(0.5, 1.5, 6) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [0.5, y, z]
    for y in torch.linspace(0.5, 1.5, 7) for z in torch.linspace(-0.5, 0.5, 6)
] + [
    [x, y, -0.5]
    for x in torch.linspace(-0.5, 0.5, 6) for y in torch.linspace(0.5, 1.5, 7)
] + [
    [x, y, 0.5]
    for x in torch.linspace(-0.5, 0.5, 6) for y in torch.linspace(0.5, 1.5, 7)
]

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

obstacles = mur_a_passer
