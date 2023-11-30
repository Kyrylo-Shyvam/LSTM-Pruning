# Project Proposal

# Deliverables:

- Create an LSTM model for NMT used in our paper.

[Compression of Neural Machine Translation Models via Pruning](https://arxiv.org/abs/1606.09274)

- The model used in paper comes from a previous work (given below). The model introduced different type of attention simpler than **Bahdanau Attention.**

[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

- Train it, (write a training code and complete training).
- Implement pruning provided in paper, given a trained model. The pruning performed is a magnitude based method that prunes the weights with smallest magnitude. The reasoning is that they have little effect on final probability distribution. We also try different common variations of magnitude-based pruning (layer wise pruning, random pruning, etc).
- Experimental report replicating ablation studies (mentioned in details later).
- Implement more recent paper on pruning for comparisons:
    
    [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340v2)
    
- ******************************************************The previously mentioned paper is for CNNs or RNN in other tasks, we want to apply similar methods for LSTMs in NMT task.****************************************************** The pruning performed here is gradient based pruning done before training procedure.
- Perform similar ablations as in the main paper.
- Create visualizion of pruning algos (how pruning is performed).
    - Make visualisations for neuronal pruning, magnitude-based pruning, etc.

# Datasets:

We will choose the dataset on the basis of computations available

- WMT'14 English-German data (4.5M sentence pairs)
- IWSLT'15 English-Vietnamese data (133K sentence pairs)
- IWSLT 2014 dataset German-English (150K  training sentences).

# Ablations/Experiments:

- Effect on BLUE score by different pruning schemes and percentage of parameters pruned (class-blind, class-uniform, class-distribution).
- Performance of pruned models after pruning, or after pruning and retraining.
- Breakdown of perplexity increase by weight class

# Timeline:

- Till 20th October
    - Read the papers with which we are going to compare
- Till 30th October(checkpoint 2)
    - Create a LSTM model for NMT
    - Train It
    - Start implementing pruning provided in the original paper
- Till 15 Nov
    - Perform experiments replicating ablation studies in the given paper
    - Start working on comparisons.
- Till 30 Nov (checkpoint 3)
    - Perform ablations with other paper
    - Form a visualisation for pruning
