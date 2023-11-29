from torch import nn
from vocab import Vocab
import graphviz
from torchview import *
import lstmModel
data="data/vocab.json"
vocab = Vocab.load(data)
model=lstmModel.NMT(embed_size=256,hidden_size=256, dropout_rate=0.2, vocab=vocab)

# Ensure the model is in evaluation mode
model.eval()

# Sample input
sample_src = [["I", "like", "pizza."]]
sample_tgt = [["<s>", "Je", "aime", "la", "pizza.", "</s>"]]

# Convert input to tensor
src_sents_var = vocab.src.to_input_tensor(sample_src, device=model.device)
tgt_sents_var = vocab.tgt.to_input_tensor(sample_tgt, device=model.device)
src_sents_len = [len(s) for s in sample_src]

# Forward pass for visualization
output, intermediate_outputs = model(src_sents_var, tgt_sents_var)

# Extract intermediate outputs for visualization
src_encodings = intermediate_outputs['src_encodings']
decoder_init_vec = intermediate_outputs['decoder_init_vec']
src_sent_masks = intermediate_outputs['src_sent_masks']
att_vecs = intermediate_outputs['att_vecs']
tgt_words_log_prob = intermediate_outputs['tgt_words_log_prob']
tgt_gold_words_log_prob = intermediate_outputs['tgt_gold_words_log_prob']
scores = intermediate_outputs['scores']

# Make the computation graph
graph = make_dot((scores, intermediate_outputs), params=dict(model.named_parameters()))
graph.render(filename='NMT_computation_graph', format='png', cleanup=True)
