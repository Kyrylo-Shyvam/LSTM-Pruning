# Documentation
- [Documentation](#documentation)
  - [Folder Structure](#folder-structure)
  - [Pruning Methods Implemented](#pruning-methods-implemented)
  - [Working Of Different Scripts](#working-of-different-scripts)
  - [Experiments Performed](#experiments-performed)

## Folder Structure

The project has the following folder structure:

- `animation/`: Contains videos of animations.
- `docs/`: Contains docs for the project.
- `code/`: Contains the whole source code for the project.
  - `data/`: Contains the data for the project.
  - `work_dir/`: Contains the models for the project.
  - `scripts/`: Contains scripts for running differnt components of the project.
  - `lstmModel.py`: Contains the LSTM model.
  - `pruningMethods.py`: Contains the pruning methods for SNIP and OBD.
- `images/`: Contains graphs for various experiments.
- `notebooks/`: Contains the notebooks used for running experiments.

## Pruning Methods Implemented
* Magnitude Pruning
  * class-blind magnitude pruning
  * class-uniform magnitude pruning
  * class-distribution magnitude pruning
* Random Pruning
* SNIP Pruning
* OBD Pruning

## Working Of Different Scripts
- `decode.sh`: Used for getting the bleu score of the model.
  - usage: `./decode.sh <model_name> <test_source_file> <test_target_file>`
  - test_source_file and test_target_file have been set up in the script they have to passed as arguments only if they are changed.
- `download.sh`: for downloading the data.
- `prune.sh`: for pruning without retraining.
  - usage: `./prune.sh <model_name> <pruning_method> <pruning_percentage>`
  - pruned model is saved to work_dir/model_name.pruned
- `retrain.sh`: for pruning with retraining.
  - usage: `./prune_retrain.sh <model_name> <pruning_method> <pruning_percentage>`
- `train.sh`: for training the model.
  - usage: `./train.sh`
  - model is saved to work_dir/model.bin
- `sparse-train.sh`:prune the model without training and then train the model.
  - usage: `./sparse-train.sh <model_name> <pruning_method> <pruning_percentage>`
- `snip-prune.sh`: pruning snip without training
  - usage: `./snip-prune.sh <pruning_percentage> <pretrain_batches>`
  - model is saved to model.bin.pruned
- `snip-train.sh`: For training the model with SNIP pruning
  - usage: `./snip-train.sh <pruning_percentage> <pretrain_batches>`
  - model is saved to work_dir/model.bin.final

## Experiments Performed
- Pruning after training without retraining and finding bleu score for differnt percentage of pruning
  - Magnitude Pruning
    - class-blind magnitude pruning
    - class-uniform magnitude pruning
    - class-distribution magnitude pruning
  - Random Pruning
  - OBD Pruning 
- Pruning with retraining and finding bleu score for differnt percentage of pruning
  - Magnitude Pruning
    - class-blind magnitude pruning
  - Random Pruning
  - OBD Pruning  
- Pruning with SNIP and finding bleu score for different percentage of training
- Pruning on extreme percetages
  - OBD
  - class-blind magnitude pruning
  - random
- Images of weights in class blind maginitude pruning
- Change in perplexity by pruning different weights classes at 90 percent pruning for class blind ,class uniform and class distrubution magnitude pruning
- Relation between perplexity change and magniture of maxinum wieght deleted for class blind magnitude pruning
- Pruning then again initialsing the weights and then training the model(Sparse-From-Begin)
- Percentage pruned per layer 
  - OBD
  - class-blind magnitude pruning
  - SNIP
- Effect Of how much data we show before pruning in snip