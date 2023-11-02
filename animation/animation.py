from manim import *
from manim_ml.neural_network.layers import FeedForwardLayer
from manim_ml.neural_network.neural_network import NeuralNetwork

class NeuralNetworkPruning(Scene):
    def construct(self):

        # Create a neural network with 3 layers: 5 nodes in the input layer, 3 nodes in the hidden layer, and 1 node in the output layer
        nn = NeuralNetwork([FeedForwardLayer(5), FeedForwardLayer(3), FeedForwardLayer(1)])

        # Add the neural network to the scene
        self.add(nn)

        # Create a list to store the pruned nodes
        pruned_nodes = []

        # Iterate over the layers of the neural network and prune the nodes with the lowest weights
        for layer in nn.layers:
            for node in layer.nodes:
                if node.weight < 0.1:
                    pruned_nodes.append(node)

        # Animate the pruning of the nodes
        for node in pruned_nodes:
            self.play(FadeOut(node.circle))

        # Display a message at the end of the animation
        self.add(Text("Neural network pruning complete!"))

# Render the animation
NeuralNetworkPruning().render()
