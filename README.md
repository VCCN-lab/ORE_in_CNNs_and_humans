Code accompanying the paper:
Diverse Visual Experience Promotes Integrated Representations and Mitigates Bias in Deep Neural Networks 
for Face Perception
E Akbari, K Dobs - bioRxiv, 2025

This repository contains the code to reproduce the analyses for the paper above where the Other-Race Effect
in VGG16 was probed and compared to human behavior.

## Repository structure
```text
.
├── activations/              # Contain the activation of the VGG16 for the test stimuli extracted from penultimate convolutional layer
├── experiment_data/          # Anonymized data from human behavior (task: triplet matching task for face identities)
├── figures/                  # Figures that are used in the paper
├── lesioning/                # Results of the lesioning analysis to find the contribution of each unit to face identification based on the paper
├── utils/                    # Helper functions that are used in the plots.ipynb to reproduce the plots of the paper
└── plots.ipynb               # Jupyter notebook file to run and get the plots
└── LICENSE                   # License terms under which the code and data of this repo can be used
└── requirements.txt          # List of the packages you need to run the analyses
└── README.md
```

## Contact

For questions, please contact:
Elaheh Akbari (elahakbri@gmail.com)
