# PSC-drones-control-with-GAN

## Warning

Not up to date yet

## How to use 

### Train the model (Example)

```bash
python3 main.py "train" "model_path" 1 0.0001 500 70 1 1 1 1 1
```

### Test the model & visualize (Example)

```bash
python3 main.py "load" "model_path" 1 0.0001 500 70 1 1 1 1 1
```

\texttt{TOTAL\_TIME} & \\
        \texttt{VARIANCE} &  \\
        \texttt{EPSILON} & \\
        \texttt{EXP} & \\
        \texttt{ALPHA\_LOSS\_G\_TERMS} & \\
        \texttt{ALPHA\_TARGET} & \\
        \texttt{ALPHA\_FORMATION} & \\
        \texttt{ALPHA\_OBSTACLE} & \\
        \texttt{ALPHA\_COLLISION} & \\
        \texttt{ALPHA\_GRAD\_PHI} & \\
        \texttt{F\_FORMATION} & \\
        \texttt{F\_FORMATION\_NAME} & \\
        \texttt{NB\_DRONES} & \\
        \texttt{CHOSEN\_INITIAL\_FORMATION} & \\
        \texttt{CHOSEN\_FINAL\_FORMATION} & \\
        \texttt{ENVIRONMENT} & \\