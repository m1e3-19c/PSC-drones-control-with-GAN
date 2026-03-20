# PSC drones control with GAN


## How to use 

### Train the model (Example)

```bash
# Give parameters as command line arguments 
python3 main.py "train" "model_path" 1 0.0001 500 70 1 1 1 1 1

# Or you can set the parameters directly in the code (main.py) and run without arguments
python3 main.py "train" "model_path"
```

### Test the model & visualize (Example)

```bash
# Give parameters as command line arguments 
python3 main.py "load" "model_path" 1 0.0001 500 70 1 1 1 1 1

# Or you can set the parameters directly in the code (main.py) and run without arguments
python3 main.py "load" "model_path"
```