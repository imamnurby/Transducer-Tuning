This directory contains the following files.
```
├── compute_metrics_per_instance.py  # Script to compute metrics for each instance
├── config                           # Contains definitions of training, inference, and model configurations
├── experiments                      # Folder containing experimental settings
│   └── code_translation
│       ├── codet5p-220m
│       ├── codet5p-770m
│       └── generation.yml
├── inference.py                     # Script for running inference
├── metrics                          # Contains definitions of BLEU and CodeBLEU metrics
├── modelling                        # Contains definitions for transducer tuning and its variants
├── README.md
├── scripts
│   ├── run_compute_metrics.sh       # Script to compute final metrics end-to-end
│   └── run_training_inference.sh    # Script to run training and inference end-to-end
├── train.py                         # Contains the training loop definition
└── utils.py                         # Contains helper functions
```

# Running Experiments
1. Set Up the Experiment Directory
   We've provided an example for code translation in the `experiments/code_translation` directory. To set up your experiment, create a folder with a similar structure and corresponding `.yml` files. There are two `.yml` files you need to create:
   - `model.yml`: Contains the model configuration.
   - `generation.yml`: Contains the configuration for inference.

2. Run Training and Inference 
   Execute `scripts/run_training_inference.sh` to run the training and inference processes. You can configure various hyperparameters through this script. We've added comments and used meaningful naming conventions to make the script easier to understand.

3. Compute Final Metrics
   After training and inference, run `scripts/run_compute_metrics.sh` to compute the final metrics before proceeding to the analysis stage.

