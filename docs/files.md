## Documentation for different files

- [Documentation for different files](#documentation-for-different-files)
- [Vocab.py](#vocabpy)
  - [Overview](#overview)
  - [Methods](#methods)
  - [Building a Vocabulary](#building-a-vocabulary)
  - [Saving and Loading a Vocabulary](#saving-and-loading-a-vocabulary)
- [Utils.py](#utilspy)
  - [`read_corpus`](#read_corpus)
  - [`batch_iter`](#batch_iter)
  - [`LabelSmoothingLoss`](#labelsmoothingloss)
- [pruningMethods.py](#pruningmethodspy)
  - [`get_layers`](#get_layers)
  - [`SNIP`](#snip)
  - [`OBD`](#obd)
- [lstmModel.py](#lstmmodelpy)
- [Pruning Functions](#pruning-functions)
  - [Random Pruning](#random-pruning)
  - [Magnitude Pruning](#magnitude-pruning)

## Vocab.py

### Overview

The `Vocab` class represents a vocabulary for a natural language. It contains two main components:

* A source vocabulary: A mapping from source words to their corresponding IDs.
* A target vocabulary: A mapping from target words to their corresponding IDs.

### Methods

The `Vocab` class provides several methods for working with vocabularies:

* `to_input_tensor(sents: List[List[str]], device: torch.device) -> torch.Tensor`: Converts a list of source sentences to a tensor of input IDs.
* `words2indices(sents): List[List[int]]`: Converts a list of source or target sentences to a list of lists of word IDs.
* `indices2words(word_ids): List[str]`: Converts a list of word IDs to a list of words.

### Building a Vocabulary

To build a vocabulary, you can use the `Vocab.build(src_sents, tgt_sents, vocab_size, freq_cutoff)` method. This method takes the following arguments:

* `src_sents`: A list of source sentences.
* `tgt_sents`: A list of target sentences.
* `vocab_size`: The maximum number of words to include in the vocabulary.
* `freq_cutoff`: The minimum frequency for a word to be included in the vocabulary.

### Saving and Loading a Vocabulary

To save a vocabulary, you can use the `Vocab.save(file_path)` method. This method takes the path to the file where the vocabulary should be saved.

To load a vocabulary, you can use the `Vocab.load(file_path)` method. This method takes the path to the file where the vocabulary was saved.

## Utils.py

### `read_corpus`

**Function:** Reads a corpus from a file.

**Parameters:**

- `file_path`: The path to the corpus file.
- `source`: The source of the corpus ('src' or 'tgt').

**Returns:**

A list of sentences.

### `batch_iter`

**Function:** Yields batches of data from a corpus.

**Parameters:**

- `data`: A list of sentences.
- `batch_size`: The batch size.
- `shuffle`: Whether to shuffle the data.

**Yields:**

A tuple of (source sentences, target sentences).

### `LabelSmoothingLoss`

**Class:** Implements the Label Smoothing Loss.

**Parameters:**

- `label_smoothing`: The label smoothing factor.
- `tgt_vocab_size`: The target vocabulary size.
- `padding_idx`: The padding index.

**Methods:**

- `forward()`: Calculates the label smoothing loss.

**Input:**

- `output`: The model output.
- `target`: The target labels.

**Output:**

A tensor of loss values.

## pruningMethods.py

### `get_layers`

**Function:** Retrieves the layers and their corresponding weights from a given model.

**Parameters:**

- `model`: The model to extract layers from.

**Returns:**

A list of tuples, where each tuple contains a layer and its corresponding weight tensor.

### `SNIP`

**Class:** Implements the SNIP (Sensitivity Pruning) algorithm.

**Methods:**

- `get_mask(model)`: Generates a pruning mask based on the sensitivity of each weight in the model.
- `set_grad(model)`: Enables the gradient computation for all weights in the model.
- `set_grad_back(model)`: Disables the gradient computation for all weights and returns the sensitivity scores for each weight.
- `thresholding(model, percent)`: Thresholds the sensitivity scores to retain a specified percentage of weights.
- `prune(model, data, batches, batch_size, percent, device)`: Performs the SNIP pruning process on the given model.

**Attributes:**

- `thresh`: The threshold value used for pruning.
- `scores`: The sensitivity scores for each weight.

### `OBD`

**Class:** Implements the OBD (Optimization-Based Dropout) algorithm.

**Methods:**

- `get_mask(model)`: Generates a pruning mask based on the Hessian of each weight in the model.
- `set_grad_back(model)`: Returns the Hessian scores for each weight.
- `thresholding(model, percent)`: Thresholds the Hessian scores to retain a specified percentage of weights.
- `prune(model, data, batches, batch_size, percent, device)`: Performs the OBD pruning process on the given model.

**Attributes:**

- `thresh`: The threshold value used for pruning.
- `scores`: The Hessian scores for each weight.

## lstmModel.py

## Pruning Functions

The provided code implements various pruning algorithms for neural network models. These algorithms can be used to reduce the size and complexity of neural networks, potentially improving their efficiency and performance.

### Random Pruning

- `random_pruning(model, percentage)`: Performs random pruning on a given model, removing a specified percentage of weights randomly.

- `random_layerwise_pruning(model, percentage)`: Applies random pruning to a given model on a layer-wise basis, removing a specified percentage of weights from each layer.

### Magnitude Pruning

- `class_blind_pruning(model, percentage)`: Performs pruning based on the L1-norm of weights, removing connections with less impact on the model's performance.

- `class_uniform_sub(module, percentage)`: Applies pruning based on the L1-norm within a module, retaining the top `percentage` weights and discarding the rest.

- `class_uniform_pruning(model, percentage)`: Applies pruning based on the L1-norm to each child module of the model.

- `class_distribution_sub(module, lamb)`: Applies pruning based on the L1-norm and a threshold parameter, retaining weights with absolute values greater than `lamb*std`.

- `class_distribution_pruning(model, lamb)`: Applies pruning based on the L1-norm and a threshold parameter to each child module of the model.


