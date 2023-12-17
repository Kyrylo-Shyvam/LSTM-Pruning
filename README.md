# Pruning of LSTMs for Neural Machine Translation

## Why do we do Pruning?

- Neural networks are often **over-parameterized**, meaning that they have more parameters than they actually need to learn the desired task. This means that many of the weights in the network are not actually doing anything useful. Pruning can help to remove these redundant weights, making the network more efficient.
- Neural networks are often trained using stochastic gradient descent (SGD). SGD can sometimes get stuck in local minima, which can lead to **suboptimal** performance. Pruning can help to prevent SGD from getting stuck in **local** **minima** by encouraging the network to learn more robust features.
- Pruning can help to improve the **generalization** performance of neural networks. Generalization performance is the ability of a network to perform well on unseen data. Pruning can help to improve generalization performance by removing weights that are not important for the specific task that the network is being trained on.

## What does Pruning Do?

During pruning, connections associated with less important or redundant features are removed, leading to a sparser neural network. This can remove the connections based on various saliency criterion or some other methods. So pruning kind of works like feature selection i.e. selecting features most important for the task.

## How does retraining help?

During pruning since we remove the connections our accuracy will decrease. Now by retraining the network is encouraged to rely on the remaining important features and adjust their weights to capture the relevant patterns in the data.

## What is the difference between structured and unstructured pruning?

- Structured pruning involves pruning entire neuron, filters etc.
- Unstructured Pruning involves pruning individual parameters based on some saliency criteria.

What we have done through our whole study is unstructured pruning

## Our Task

The task which we are working on is **Neural Machine Translation.** It involves translating a sentence X of one language(source language) to sentence y of another language(target language).

********************************************************Encoder Decoder Architecture:********************************************************

An encoder computes representations for each source sentence.Decoder generates one target word at a time.

******************Alignment******************: Word level correspondence between source sentence x and target sentence y.

What we calculate in ******NMT****** is:

$$p(y \mid x) = p(y_1 \mid s) \cdot p(y_2 \mid y_1, s) \cdot \ldots \cdot p(y_T \mid y_{T-1}, y_{T-2}, \ldots, y_1, s)$$

********************************************Beam Search Decoding:******************************************** On each decoding step we keep track of k most probable translations (which we call hypothesis).

**********************************Word Embeddings: W**********************************ord in sentence are converted into vectors such that these vectors are learned to capture context and similar words used in similar context have more close word embeddings. Basically we get the features of the words in the form of numbers.

## Evaluation Measures

1. **Perplexity:**

Intrinsic measurement of how good a model is. A good model is one which gives high probability to the correct sentence

Perplexity is exponential of average negative log likelihood. Decreasing perplexity is equivalent to increasing probability. Suppose we have 10 items with equal probability then perplexity is 10 showing that the model cannot choose among those 10 and is confused.

$$
Perplexity = \frac{1}{{\sqrt[N]{P(w_1, w_2, \ldots, w_N)}}}
$$

$$
Perplexity = \frac{1}{{\sqrt[N]{\prod_{i=1}^N p(w_i)}}}
$$

After taking log and simplifying we get:

$$
\text{Perplexity} = e^{-\Sigma_{i=1}^{N} \log_e(P(w_i)) / N}
$$

1. **Bleu Score**

Given a machine translation it tells how good is the machine translation.It indicates how similar the candidate text is to the reference texts that are provided in the dev or test set.

![Screenshot 2023-11-02 at 3.31.06 PM.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Screenshot_2023-11-02_at_3.31.06_PM.png)

![Screenshot 2023-11-02 at 3.36.47 PM.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Screenshot_2023-11-02_at_3.36.47_PM.png)

- **Brevity Penalty:**

It penalises very short candidates as almost all the words in candidate can be present in reference if the candidate is very short compared to reference.

- **Disadvantages of Bleu Score:**
    - Not good at capturing meaning and grammar of a sentence.  does not consider long range dependencies.
    - No difference between content and function words
    - The BLEU metric performs badly when used to evaluate individual sentences as it is a corpus based metric.
    - It does not take into consideration order of the n-grams in the reference and candidate sentence

## Why does Pruning weights below a threshold work?

During pruning we prune weights that are redundant for our network. Weights with less magnitude are redundant because they have a smaller impact on the output of the network. This is because neural networks use a non-linear activation function, such as the sigmoid or ReLU function, after each layer of weights. These activation functions squash the input values into a smaller range, which has the effect of attenuating the impact of small weights.

## Three types of pruning schemes we implemented:

- **Class Blind:**
    - Sort the parameters of whole model. Find the threshold for pruning such that x% of weights is pruned.
      
    [![ExampleScene.mp4](https://img.youtube.com/vi/G0qQOhCKzt8/0.jpg)](https://youtu.be/G0qQOhCKzt8)
    
- **Class Uniform:**
    - Sort the parameters of  each class and within each class find the threshold for pruning such that x% of weights in all classes of weights are pruned.
      
[![ExampleScene.mp4](https://img.youtube.com/vi/-bwvUtaBjnE/0.jpg)](https://youtu.be/-bwvUtaBjnE)

    
- **Class Distribution:**
    - For each class c, weights with magnitude less than $\lambda \sigma_c$  are pruned. Here, $\sigma_c$ is the standard deviation of that class and $\lambda$ is a universal parameter chosen such that in total, x% of all parameters are pruned.

## Comparison of these three pruning schemes:

We can see that class-blind performs better than both class-uniform and class-distribution. 

![different_pruning_schemes4.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/different_pruning_schemes4.png)

- **Random Global Pruning -** x% of weights are randomly pruned.

Magnitude pruning works better than random pruning, making it clear that weights with less magnitude are more likely to be redundant. All 3 kinds of magnitude pruning outperform random pruning.

## Comparison Of Retraining And Sparse From Begin Models:

Comparison of random pruning coupled with retraining is included.

![retraining_with_random.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/retraining_with_random.png)

Graph shows that retraining is better, and it makes even random pruning much better. We can also use the sparse structure obtained during pruning to train sparse model from scratch. As we can see it is worse than retraining.

## Percentage Pruned Per Class of Weights in Class-blind and Class-distribution for 90 percent pruning:

![classBlindPercentagePrunedPerClass.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/classBlindPercentagePrunedPerClass.png)

![classDistributionPercentagePrunedPerClass.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/classDistributionPercentagePrunedPerClass.png)

## Perplexity Change with pruning of each class of weights:

![bar-graph.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/bar-graph.png)

The graph shows that that some layers give higher perplexity change even though same number is pruned. This means those layers are more important and pruning them equally is not a good strategy. Interestingly last layers (softmax and attention) are more important than initial (target and source embeddings).

## Change in perplexity vs max weight removed of that class

![devPplVsMaxWeight.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/devPplVsMaxWeight.png)

## Encoder and Decoder Weights Visualisation

**a) Encoder:**

![Encoder.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Encoder.png)

<Explanation>

**b) Decoder:**

![Decoder.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Decoder.png)

<Explanation>

## Median Bleu Score Comparison:

![median_prunedType.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/median_prunedType.png)

## Comparison with other Techniques:

## Optimal Brain Damage

It a gradient pruning technique by LeCun et. al published in 1989 which started the field of pruning neural network weights.

Using Taylor’s expansion we can approximate change in loss, $d L$ as: 

$$
\begin{equation}
\begin{split}
\text{dL} = &  \sum_{i} \frac{\partial L}{\partial w_i } w_i +  \frac{1}{2} \sum_i \left( \frac{\partial^2 L}{\partial w_i^2} \right) \left( \frac{\partial L}{\partial w_i} \right)^2+\frac{1}{2} \sum_{i,j} \frac{\partial^2 L}{\partial w_i \partial w_j} w_i w_j + O(n^3)
\end{split}
\end{equation}
$$

Now since the model has converged, the first term is zero and we ignore the interaction and higher order terms what we finally get is:

$$
\begin{equation}
\begin{split}
\text{dL} = &\sum_i \left( \frac{\partial^2 L}{\partial w_i^2} \right) \left( \frac{\partial L}{\partial w_i} \right)^2 
\end{split}
\end{equation}
$$

Second derivative of the loss functions with respect to parameters is used a saliency measure. 
[![ExampleScene.mp4](https://img.youtube.com/vi/wT6mGmLONeg/0.jpg)](https://youtu.be/wT6mGmLONeg)

### Why does this work?

Gradient pruning prunes the weights gradients with respect to whose are negligible as these weights do not cause more change the loss function so they must not be that important.

### Effect of extreme pruning percentages on OBD:

![The effect of extreme pruning  on OBD.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/The_effect_of_extreme_pruning__on_OBD.png)

## Single Shot Network Pruning (SNIP) based on connection sensitivity:

### Why is even there a need for pruning before training?

- It identifies saliency criterion based on connection sensitivity that identifies structurally important connections in the network for the given task before training starts.
- This eliminates the need for both retraining and the complex pruning schedule while making it *robust to architecture variations.*
    
    ![Untitled](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Untitled.png)
    
    **Saliency criterion**:  Connection sensitivity involves removing the connection and observing the change in the network's output (loss function value) compared to the original network. So it does not matter if the loss has converged, we want only to see how much change does removing a weight produce to loss.
    

It can be modeled in following way:

Here, $c$ is the mask which says if the weight must be present or not. $c$ is element wise multiplied with corresponding weight. 

Ideally, we would want to remove each $w_j$ once (by setting $c_j = 0)$, to check for effect on loss:

 

![Untitled](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Untitled%201.png)

However, we cannot do that in reasonable time, so we approximate the effect by calculating $\frac{\delta L}{\delta c}$:

$$
\frac{\delta L}{\delta c}
$$

![Untitled](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Untitled%202.png)

Note that it is gradient with respect to mask, and not original weights differentiating it with previous methods.

It involves pruning the network once at initialisation and then training normally. 

************************************************Effect of initialisation:************************************************

Different initialisation tried are:

- Uniform initialisations (as recommended by authors for given LSTM).
- Kaiming uniform
- Kaiming normal
- Xavier uniform
- Xavier normal

![Screenshot 2023-11-29 at 5.50.28 AM.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Screenshot_2023-11-29_at_5.50.28_AM.png)

Initialisation matters a lot since the connection sensitivity is performed on untrained weights.. It should be variance scaled for the method to not depend on architecture variations.

### Effect of batch size to find saliency -

In SNIP results might also change with how much data we show, before actually calculating the gradients to find saliency.

![The effect of different batch sizes.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/The_effect_of_different_batch_sizes.png)

The graph shows that just $\frac{50^{th}}{1000}$ portion of dataset is enough to make a good predictions on pruning weights.

### Problems with SNIP:

- Connection sensitivity is sub-optimal as a criterion because the gradient of each weight might change dramatically after pruning due to complicated interactions between weights.
- Since SNIP only considers the gradient for one weight in isolation, it could remove connections that are important to the flow of information through the network.

## Different Experiment Performed For Comparison

### Percentage Pruned vs class type pruned for 50 and 80 percent pruning

![Percentage pruned vs type of pruning-50.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Percentage_pruned_vs_type_of_pruning-50.png)

![Percentage pruned vs type of pruning-80.png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/Percentage_pruned_vs_type_of_pruning-80.png)

OBD and SNIP remove weights closer to start. This is actually good as later weights are considered to be more important.

### Effect of extreme pruning percentages comparison

**a) Without retraining:**

![The effect of extreme pruning (normal).png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/The_effect_of_extreme_pruning_(normal).png)

SNIP was pruned at initialisation, rest are pruned after training.

**b) With retraining:**

![The effect of extreme pruning (retrained).png](Documentation%20630d93bc52214cb5aedfc667b5e5554e/The_effect_of_extreme_pruning_(retrained).png)

Top-3 pruning schemes are shown after retraining. SNIP is shown as usual. 

Both graphs seem to confirm that SNIP is good at extreme levels of pruning.
