import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from Model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def extract(g):
    global xg  # derivatives of middle variants
    xg = g


class Semantic_Discriminator(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, layer_dropout=0.4):
        super(Semantic_Discriminator, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=(0 if num_layers == 1 else layer_dropout), bidirectional=False, batch_first=True)

        self.kernel_size = 2  # means capture 2-gram features
        self.bias = nn.Parameter(torch.randn(1, hidden_size))
        self.hidden_size = hidden_size
        self.logits_dense = nn.Linear(int(self.hidden_size / 2), 1)


        self.dense = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.ReLU()
        )


    def conv2d(self, input, output_dim, k_h, k_w):
        '''
        filter: [filter_height, filter_width, in_channels, out_channels]
        :param input: (N, C_{in}, H_{in}, W_{in}) e.g., (5, 512, 1, 48)
        :param output_dim: self.hidden_size
        :param k_h: 1
        :param k_w: kernel_size
        :return:
        '''
        input = input.squeeze(1).transpose(1, 2).unsqueeze(2)
        convolve = nn.Conv2d(input.size(1), output_dim, kernel_size=(1, self.kernel_size), stride=(1, 1), padding=0).cuda()
        return convolve(input)  # nn.Conv2d: bias=true


    def SemanticClassify(self, data_pair, feedback, context, batch_norm=False):
        '''
        :param data_pair: (bsz, y_len, hz)
        :param feedback: (bsz, hz)
        :param context: (bsz, hz)
        :param emotion_context: (bsz, hz)
        :param batch_norm: bool
        :return:
        '''
        conv = self.conv2d(data_pair.unsqueeze(1), self.hidden_size, 1, self.kernel_size)

        if batch_norm:
            conv_norm = nn.BatchNorm2d(self.hidden_size).cuda()  # Applies Batch Normalization over a 4D input (a mini-batch of 2D input with additional channel dimension)
            conv = conv_norm(conv)
        # kernel_size, stride, padding, dilation
        conv_RelU = nn.ReLU()
        conv = conv_RelU(conv)

        max_pool = nn.MaxPool2d(kernel_size=(1, data_pair.size(1)-self.kernel_size+1), stride=(1, 1), padding=0)

        pooled = max_pool(conv)  #  (bsz,1,H_out,W_out)
        pooled = torch.squeeze(pooled)  # fixed vector
        # combibe data(pooled), context, and emotional context
        fused_ReLu = nn.ReLU()
        fused = fused_ReLu(pooled + context + feedback + self.bias)

        dense_fused = self.dense(fused)  # (bsz, int(hz/2))
        if batch_norm:
            dense_fused_norm = nn.BatchNorm2d(int(self.hidden_size/2)).cuda()
            # Input: :math:`(N, C, H, W)` Output: :math:`(N, C, H, W)` (same shape as input)
            dense_fused = dense_fused.unsqueeze(2)
            dense_fused = dense_fused.unsqueeze(3)  # (bsz, hsz/2, 1, 1)
            dense_fused = dense_fused_norm(dense_fused)
            dense_fused = dense_fused.squeeze(2)
            dense_fused = dense_fused.squeeze(2)
        logits = self.logits_dense(dense_fused)  # (bsz, 1)
        return logits


    def forward(self, neg, pos, fdb, context, emotion_context=None):
        # 1 LSTM  - negative prediction
        embed_neg = self.embedding(neg)
        outputs_neg, hidden_neg = self.lstm(embed_neg)
        last_hidden_neg = hidden_neg[1][-1]

        # 2 LSTM  - positive response
        embed_pos = self.embedding(pos)
        outputs_pos, hidden_pos = self.lstm(embed_pos)
        last_hidden_pos = hidden_pos[1][-1]

        # 3 LSTM  - next feedback
        embed_fdb = self.embedding(fdb)
        outputs_fdb, hidden_fdb = self.lstm(embed_fdb)
        last_hidden_fdb = hidden_fdb[1][-1]

        neg_sample = outputs_neg
        pos_sample = outputs_pos

        # 4 semantic classify
        neg_semantic_logits = self.SemanticClassify(neg_sample, last_hidden_fdb, context, batch_norm=True)
        pos_semantic_logits = self.SemanticClassify(pos_sample, last_hidden_fdb, context, batch_norm=True)

        # 5 discriminator_loss
        disc_sem_loss = torch.mean(neg_semantic_logits - pos_semantic_logits)
        gen_sem_loss = torch.mean(neg_semantic_logits)


        # 6 wgan
        alpha_empty = torch.empty(context.size(0), 1, 1)  # (bsz, 1, 1)
        alpha = Variable(torch.nn.init.uniform_(tensor=alpha_empty, a=0., b=1.)).cuda()  # alpha~[0,1]
        interpolates = alpha * neg_sample + ((1 - alpha) * pos_sample)
        disc_interpolates = self.SemanticClassify(interpolates, context, True)

        interpolates.register_hook(extract)
        disc_interpolates.backward(torch.ones_like(disc_interpolates), retain_graph=True)
        gradients = xg  # normalization

        # two norm
        slopes = torch.sqrt_(torch.sum(torch.mul(gradients, gradients), 1))
        gradient_penalty = torch.mean((slopes-1) ** 2)
        gradient_penalty = config.gp_lambda * gradient_penalty
        assert torch.sum(torch.isnan(gradient_penalty)) == 0, "omg!!!!"
        disc_sem_loss += gradient_penalty  # add gradient norm

        return disc_sem_loss, gen_sem_loss


class Emotional_Discriminator(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, layer_dropout=0.4):
        super(Emotional_Discriminator, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=(0 if num_layers == 1 else layer_dropout), bidirectional=False, batch_first=True)

        self.kernel_size = 2  # capture 2-gram features
        self.bias = nn.Parameter(torch.randn(1, hidden_size))
        self.hidden_size = hidden_size
        self.logits_dense = nn.Linear(int(self.hidden_size / 2), 1)


        self.dense = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.ReLU()
        )


    def conv2d(self, input, output_dim, k_h, k_w):
        '''
        filter: [filter_height, filter_width, in_channels, out_channels]
        :param input: (N, C_{in}, H_{in}, W_{in}) e.g., (5, 512, 1, 48)
        :param output_dim: self.hidden_size
        :param k_h: 1
        :param k_w: kernel_size
        :return:
        '''
        input = input.squeeze(1).transpose(1, 2).unsqueeze(2)
        convolve = nn.Conv2d(input.size(1), output_dim, kernel_size=(1, self.kernel_size), stride=(1, 1), padding=0).cuda()
        return convolve(input)


    def EmotionClassify(self, data_pair, feedback, context, emotion_context, batch_norm=False):
        '''
        :param data_pair: (bsz, y_len, hz)
        :param feedback: (bsz, hz)
        :param context: (bsz, hz)
        :param emotion_context: (bsz, hz)
        :param batch_norm: bool
        :return:
        '''
        conv = self.conv2d(data_pair.unsqueeze(1), self.hidden_size, 1, self.kernel_size)  # (bsz, hsz, 1, y_len)

        if batch_norm:
            conv_norm = nn.BatchNorm2d(self.hidden_size).cuda()  # Applies Batch Normalization over a 4D input (a mini-batch of 2D input with additional channel dimension)
            conv = conv_norm(conv)
        # kernel_size, stride, padding, dilation
        conv_RelU = nn.ReLU()
        conv = conv_RelU(conv)

        max_pool = nn.MaxPool2d(kernel_size=(1, data_pair.size(1)-self.kernel_size+1), stride=(1, 1), padding=0)

        pooled = max_pool(conv)  #  (bsz,1,H_out,W_out)
        pooled = torch.squeeze(pooled)  # fixed vector
        # combibe data(pooled), context, and emotional context
        fused_ReLu = nn.ReLU()
        fused = fused_ReLu(pooled + emotion_context + feedback + self.bias)


        dense_fused = self.dense(fused)  # (bsz, int(hz/2))
        if batch_norm:
            dense_fused_norm = nn.BatchNorm2d(int(self.hidden_size/2)).cuda()
            # Input: :math:`(N, C, H, W)` Output: :math:`(N, C, H, W)` (same shape as input)
            dense_fused = dense_fused.unsqueeze(2)
            dense_fused = dense_fused.unsqueeze(3)  # (bsz, hsz/2, 1, 1)
            dense_fused = dense_fused_norm(dense_fused)
            dense_fused = dense_fused.squeeze(2)
            dense_fused = dense_fused.squeeze(2)
        logits = self.logits_dense(dense_fused)  # (bsz, 1)
        return logits


    def forward(self, neg, pos, fdb, context, emotion_context):
        # 1 LSTM  - negative prediction
        embed_neg = self.embedding(neg)
        outputs_neg, hidden_neg = self.lstm(embed_neg)
        last_hidden_neg = hidden_neg[1][-1]

        # 2 LSTM  - positive response
        embed_pos = self.embedding(pos)
        outputs_pos, hidden_pos = self.lstm(embed_pos)
        last_hidden_pos = hidden_pos[1][-1]

        # 3 LSTM  - next feedback
        embed_fdb = self.embedding(fdb)
        outputs_fdb, hidden_fdb = self.lstm(embed_fdb)
        last_hidden_fdb = hidden_fdb[1][-1]

        neg_sample = outputs_neg
        pos_sample = outputs_pos

        # 4 semantic classify
        neg_emotion_logits = self.EmotionClassify(neg_sample, last_hidden_fdb, context, emotion_context, batch_norm=True)  # (bsz, 1)
        pos_emotion_logits = self.EmotionClassify(pos_sample, last_hidden_fdb, context, emotion_context, batch_norm=True)  # (bsz, 1)

        # 5 discriminator_loss
        disc_emo_loss = torch.mean(neg_emotion_logits - pos_emotion_logits)
        gen_emo_loss = torch.mean(neg_emotion_logits)


        # 6 wgan
        alpha_empty = torch.empty(context.size(0), 1, 1)
        alpha = Variable(torch.nn.init.uniform_(tensor=alpha_empty, a=0., b=1.)).cuda()  # alpha~[0,1]
        interpolates = alpha * neg_sample + ((1 - alpha) * pos_sample)
        disc_interpolates = self.EmotionClassify(interpolates, context, emotion_context, True)  # (bs, 1)

        interpolates.register_hook(extract)
        disc_interpolates.backward(torch.ones_like(disc_interpolates), retain_graph=True)
        gradients = xg  # normalization

        # two norm
        slopes = torch.sqrt_(torch.sum(torch.mul(gradients, gradients), 1))
        gradient_penalty = torch.mean((slopes-1) ** 2)
        gradient_penalty = config.gp_lambda * gradient_penalty
        assert torch.sum(torch.isnan(gradient_penalty)) == 0, "omg!!!!"
        disc_emo_loss += gradient_penalty  # add gradient norm

        return disc_emo_loss, gen_emo_loss



class EmpDG_D(nn.Module):
    def __init__(self, vocab, model_file_path=None, is_eval=False, load_optim=False):
        '''
        the implementation of discriminators.
        '''
        super(EmpDG_D, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.sem_disc = Semantic_Discriminator(vocab, config.emb_dim, config.rnn_hidden_dim, num_layers=config.hop,)
        self.emo_disc = Emotional_Discriminator(vocab, config.emb_dim, config.rnn_hidden_dim, num_layers=config.hop)

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter, running_avg_ppl, f1_g,
                                                                                            f1_b, ent_g, ent_b))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, context, emotion_context=None, batch=None, train=False, iter_train=False):
        neg = batch["pred_batch"]
        pos = batch["target_batch"]
        fdb = batch["feedback_batch"]
        loss_d, loss_g = self.sem_disc(neg, pos, fdb, context, None)

        if config.emotion_disc:
            neg_emo = batch["pred_emotion_batch"]
            pos_emo = batch["target_emotion_batch"]
            fdb_emo = batch["feedback_emotion_batch"]
            emo_loss_d, emo_loss_g = self.emo_disc(neg_emo, pos_emo, fdb_emo, context, emotion_context)
            loss_d += emo_loss_d
            loss_g += emo_loss_g

        self.optimizer.zero_grad()
        if train:
            loss_d.backward()
            self.optimizer.step()
            for n, p in self.sem_disc.named_parameters():
                torch.clamp(p, -0.01, 0.01)
            if config.emotion_disc:
                for n, p in self.emo_disc.named_parameters():
                    torch.clamp(p, -0.01, 0.01)
        elif iter_train:
            loss_d.backward(retain_graph=True)
            self.optimizer.step()
            for n, p in self.sem_disc.named_parameters():
                torch.clamp(p, -0.01, 0.01)
            if config.emotion_disc:
                for n, p in self.emo_disc.named_parameters():
                    torch.clamp(p, -0.01, 0.01)
        else:
            print("maybe u encounter an error :( ...")

        return loss_d.item(), loss_g.item()



