#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import read_corpus, batch_iter, LabelSmoothingLoss
from vocab import Vocab, VocabEntry


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


# In[ ]:


class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, input_feed=True, label_smoothing=0.):
        super(NMT, self).__init__()
        self.embed_size = embed_size 
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab 
        self.input_feed = input_feed 
        #input feed = true is used when we want to inconporate attentional effects we want to have information not only about the last hidden layer but all hidden layers
        '''Mentioned in paper: The model will be aware of previous alignment choices.'''
        # initialize neural network layers
        #len(vocab.src) gives the no of distinct words in the source vocab
        self.src_embed = nn.Embedding(len(vocab.src), embed_size, padding_idx=vocab.src['<pad>'])#layer for generating source embedding
        self.tgt_embed = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=vocab.tgt['<pad>'])#layer for generating target embedding
        
        #we are using a bidirectional lstm as The position of specific information in sentences varies across languages due to differences in word order, grammar, and linguistic structures. 
        #this leads to more computation as we have to compute in both forward and backward direction
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True)#hidden_size is the no of neurons in the hidden layer
        
        #it takes both the current word's embedding and the previous hidden state as input to the LSTM. 
        decoder_lstm_input = embed_size + hidden_size if self.input_feed else embed_size
        self.decoder_cell_init =  nn.Linear(hidden_size * 2, hidden_size)
        
        #lstm cells constitute lstm
        self.decoder_lstm = nn.LSTMCell(decoder_lstm_input, hidden_size)#we need to use hidden state after every step for calculating the context vector
        self.pred = nn.Linear(hidden_size, len(vocab.tgt), bias=False)# prediction layer of the target vocabulary
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # bi-directional to normal size
        self.att_src_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # Projection of context(2h, context of src encodings) + hidden, to 1h attention (also incorporated to input feeding)
        self.att_vec_linear = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)
        
        self.label_smoothing = label_smoothing
        if label_smoothing > 0.:
            self.label_smoothing_loss = LabelSmoothingLoss(label_smoothing,
                                                           tgt_vocab_size=len(vocab.tgt), padding_idx=vocab.tgt['<pad>'])
            
    @property
    def device(self) -> torch.device:
        return self.src_embed.weight.device   
    
    #function for encoding the input returns the hidden states and decoder_init_state
    def encode(self, src_sents_var: torch.Tensor, src_sent_lens: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        src_sents_var is are the list of source sentence tokens
        src_sent_lens is list giving the lengths of sentences in the batch
        for eg
        Sentence 1: "I like pizza."
        Sentence 2: "She enjoys pasta."
        Assuming a simplified vocabulary where each word is represented by a unique integer:
        "I" is 1,"like" is 2,"pizza" is 3,"She" is 4,"enjoys" is 5,"pasta" is 6
        src_sents_var = torch.Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (3, 2)
        src_sent_lens = [3, 3]
        so src_sents_var is of shape (src_sent_len,batch_size)
        src_word_embeds = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        ]
        shape is (3,2,3)
        basically (src_sent_len, batch_size, embed_size)
        """
        #when we convert src_sents_var to src_words_embeds the shape is (src_sent_len, batch_size, embed_size)
        src_word_embeds = self.src_embed(src_sents_var)
        #suppose the size of differnt sentences are different then we pad the sequences otherwise batch computation would cause problems
        packed_src_embed = pack_padded_sequence(src_word_embeds, src_sent_lens)

        # src_encodings: (src_sent_len, batch_size, hidden_size * 2) this hidden_size*2 is because of using bidirectional lstm
        #last_state is the last hidden state shape->(batch_size, hidden_size*2)
        #last_cell is the last cell state shape->(batch_size,hidden_size*2)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        
        #unpack the source encodings
        src_encodings, _ = pad_packed_sequence(src_encodings)

        # (batch_size, src_sent_len, hidden_size * 2) 
        src_encodings = src_encodings.permute(1, 0, 2)
        #last_cell[0]: The final cell state of the forward LSTM, which has processed the source sentence from left to right.
        #last_cell[1]: The final cell state of the backward LSTM, which has processed the source sentence from right to left.
        # shape->(batch_size, 2 * hidden_size)
        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], dim=1))
        dec_init_state = torch.tanh(dec_init_cell)
        #src encodings are the hidden state for tokens in the input sequence
        return src_encodings, (dec_init_state, dec_init_cell)
    
    #function to remove the effect of attention from the padded values
    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1 #what ever has been padded is set to 1 in this mask

        return src_sent_masks.to(self.device)
    
    def decode(self, src_encodings: torch.Tensor, src_sent_masks: torch.Tensor,
               decoder_init_vec: Tuple[torch.Tensor, torch.Tensor], tgt_sents_var: torch.Tensor) -> torch.Tensor:
        #transforms the size of src_encoding from (src_sent_len, batch_size, hidden_size * 2) to (src_sent_len, batch_size, hidden_size)
        src_encoding_att_linear = self.att_src_linear(src_encodings)
        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = self.tgt_embed(tgt_sents_var)

        h_tm1 = decoder_init_vec

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)

                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks)

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    #function for what happens at each step of decoding 
    def step(self, x: torch.Tensor,
             h_tm1: Tuple[torch.Tensor, torch.Tensor],
             src_encodings: torch.Tensor, src_encoding_att_linear: torch.Tensor, src_sent_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        
        # h_t: (batch_size, hidden_size)
        #hidden state and cell state of lstm
        h_t, cell_t = self.decoder_lstm(x, h_tm1)
        
        #getting context vector and a_t that is the probability given to each encoder hidden state 
        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear, src_sent_masks)

        #getting attentional hidden state
        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t
    
    #function for getting the score using dot product attention
    def dot_prod_attention(self, h_t: torch.Tensor, src_encoding: torch.Tensor, src_encoding_att_linear: torch.Tensor,
                           mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, src_sent_len)
        #getting the scores by multiplying hidden layer of decoder and hidden layer of encoder all collectively
        scores = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        
        #setting scores to -infinity incase the bool mask is 1 as we set 1 mask for all the padded values
        if mask is not None:
            scores.data.masked_fill_(mask.bool(), -float('inf'))
        #getting probabilities for each hidden state
        a_t = F.softmax(scores, dim=-1)

        att_view = (scores.size(0), 1, scores.size(1))
        
        # (batch_size, hidden_size)
        #getting the context vector by weighted multiplication of encoder hidden states and a_t
        ctx_vec = torch.bmm(a_t.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, a_t
    
    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        # (src_sent_len, batch_size)
        src_sents_var = self.vocab.src.to_input_tensor(src_sents, device=self.device)
        # (tgt_sent_len, batch_size)
        tgt_sents_var = self.vocab.tgt.to_input_tensor(tgt_sents, device=self.device)
        src_sents_len = [len(s) for s in src_sents]

        src_encodings, decoder_init_vec = self.encode(src_sents_var, src_sents_len)

        src_sent_masks = self.get_attention_mask(src_encodings, src_sents_len)

        # (tgt_sent_len - 1, batch_size, hidden_size)
        att_vecs = self.decode(src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var[:-1])

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        tgt_words_log_prob = F.log_softmax(self.pred(att_vecs), dim=-1)

        if self.label_smoothing:
            # (tgt_sent_len - 1, batch_size)
            tgt_gold_words_log_prob = self.label_smoothing_loss(tgt_words_log_prob.view(-1, tgt_words_log_prob.size(-1)),
                                                                tgt_sents_var[1:].view(-1)).view(-1, len(tgt_sents))
        else:
            # (tgt_sent_len, batch_size)
            tgt_words_mask = (tgt_sents_var != self.vocab.tgt['<pad>']).float()

            # (tgt_sent_len - 1, batch_size)
            tgt_gold_words_log_prob = torch.gather(tgt_words_log_prob, index=tgt_sents_var[1:].unsqueeze(-1), dim=-1).squeeze(-1) * tgt_words_mask[1:]

        # (batch_size)
        scores = tgt_gold_words_log_prob.sum(dim=0)

        return scores


# In[ ]:


def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
    """
    Given a single source sentence, perform beam search

    Args:
        src_sent: a single tokenized source sentence
        beam_size: beam size
        max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

    Returns:
        hypotheses: a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
    """

    src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

    src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
    src_encodings_att_linear = self.att_src_linear(src_encodings)

    h_tm1 = dec_init_vec
    att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

    eos_id = self.vocab.tgt['</s>']

    hypotheses = [['<s>']]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)

        exp_src_encodings = src_encodings.expand(hyp_num,
                                                 src_encodings.size(1),
                                                 src_encodings.size(2))

        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                       src_encodings_att_linear.size(1),
                                                                       src_encodings_att_linear.size(2))

        y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
        y_tm1_embed = self.tgt_embed(y_tm1)

        if self.input_feed:
            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
        else:
            x = y_tm1_embed

        (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
                                                  exp_src_encodings, exp_src_encodings_att_linear, src_sent_masks=None)

        # log probabilities over target words
        log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

        live_hyp_num = beam_size - len(completed_hypotheses)
        contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

        prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
        hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = self.vocab.tgt.id2word[hyp_word_id]
            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
            if hyp_word == '</s>':
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                       score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
        h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
        att_tm1 = att_t[live_hyp_ids]

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                               score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses




# In[ ]:


@staticmethod
def load(model_path: str):
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    args = params['args']
    model = NMT(vocab=params['vocab'], **args)
    model.load_state_dict(params['state_dict'])

    return model

def save(self, path: str):
    print('save model parameters to [%s]' % path, file=sys.stderr)

    params = {
        'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                     input_feed=self.input_feed, label_smoothing=self.label_smoothing),
        'vocab': self.vocab,
        'state_dict': self.state_dict()
    }

    torch.save(params, path)


# In[ ]:


def evaluate_ppl(model, dev_data, batch_size=32):
    """
        perplexity is exponential of average negative log likelihood
    """

    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            # forward pass returns a vector of size batch size containing log likelihood of each sentence in the batch we sum over all the losses and multiply by -1 to get negative of log likelihood that is loss
            loss = -model(src_sents, tgt_sents).sum()
            
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict
        
        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


# In[ ]:


def train(args: Dict):
    #appending <s> and </s> to all sentences
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    #preparing train data and dev data
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])

    #defining the model
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                input_feed=args['--input-feed'],
                label_smoothing=float(args['--label-smoothing']),
                vocab=vocab)
    #switch to training mode
    model.train()

    #doing uniform initialisation we need to try other initialisatin too
    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            # (batch_size)
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                #we try out many models and take the best one intially ther is no model so first conditon is for that
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
        
                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)#so if we get worse more than patience no of times we early stop it
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)
    


# In[ ]:


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


# In[ ]:


def decode(args: Dict[str, str]):
    
    print(f"load test source sentences from [{args['TEST_SOURCE_FILE']}]", file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print(f"load test target sentences from [{args['TEST_TARGET_FILE']}]", file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))
    # get the beam sized hypothesis using beam search
    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        #get the bleu_score on test data
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


# In[ ]:


def main():
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid run mode')


if __name__ == '__main__':
    main()
