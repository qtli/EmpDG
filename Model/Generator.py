
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if (self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, pred_emotion=None, emotion_contexts=None, mask=None):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)

        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, pred_emotion=None, emotion_context=None, attn_dist=None, enc_batch_extend_vocab=None,
                max_oov_length=None, temp=1, beam_search=False, attn_dist_db=None):
        # pred_emotion (bsz, 1, embed_dim);  emotion_context: (bsz, emb_dim)
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        if emotion_context is not None:
            # emotion_context = emotion_context.unsqueeze(1).repeat(1, x.size(1), 1)
            pred_emotion = pred_emotion.repeat(1, x.size(1), 1)
            x = torch.cat((x, pred_emotion), dim=2)  # (bsz, tgt_len, 2 emb_dim)
            logit = self.emo_proj(x)
        else:
            logit = self.proj(x)  # x: (bsz, tgt_len, emb_dim)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)] * x.size(1),
                                                1)  ## extend for all seq

            # if beam_search:
            #     enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class MK_Dec(nn.Module):

    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super(MK_Dec, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)

        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

        ## GRAPH
        self.dropout = config.dropout
        self.W_q = nn.Linear(config.emb_dim, config.emb_dim)
        self.W_k = nn.Linear(config.emb_dim, config.emb_dim)
        self.W_v = nn.Linear(config.emb_dim, config.emb_dim)
        self.graph_out = nn.Linear(config.emb_dim, config.emb_dim)
        self.graph_layer_norm = LayerNorm(config.hidden_dim)

        ## emotional signal distilling
        self.identify = nn.Linear(config.emb_dim, decoder_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        ## multiple decoders
        self.emotion_embedding = nn.Linear(decoder_number, config.emb_dim)
        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000,
                                     torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

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

    def concept_graph(self, context, concept, adjacency_mask):
        '''

        :param context: (bsz, max_context_len, embed_dim)
        :param concept: (bsz, max_concept_len, embed_dim)
        :param adjacency_mask: (bsz, max_context_len, max_context_len + max_concpet_len)
        :return:
        '''
        # target = self.W_sem_emo(context)  # (bsz, max_context_len, emb_dim)
        # concept = self.W_sem_emo(concept)
        target = context
        src = torch.cat((target, concept), dim=1)  # (bsz, max_context_len + max_concept_len, emb_dim)

        # QK attention
        q = self.W_q(target)  # (bsz, tgt_len, emb_dim)
        k, v = self.W_k(src), self.W_v(src)  # (bsz, src_len, emb_dim); (bsz, src_len, emb_dim)
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2))  # batch matrix multiply (bsz, tgt_len, src_len)

        adjacency_mask = adjacency_mask.bool()
        attn_weights_ori.masked_fill_(
            adjacency_mask,
            1e-24
        )  # mask PAD
        attn_weights = torch.softmax(attn_weights_ori, dim=-1)  # (bsz, tgt_len, src_len)

        if torch.isnan(attn_weights).sum() != 0:
            pdb.set_trace()

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # weigted sum
        attn = torch.bmm(attn_weights, v)  # (bsz, tgt_len, emb_dim)
        attn = self.graph_out(attn)

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        new_context = self.graph_layer_norm(target + attn)

        new_context = torch.cat((new_context, concept), dim=1)
        return new_context

    def train_one_batch(self, batch, iter, train=True):
        enc_batch = batch["context_batch"]
        enc_batch_extend_vocab = batch["context_ext_batch"]
        enc_vad_batch = batch['context_vad']
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        concept_input = batch["concept_batch"]  # (bsz, max_concept_len)
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']
        dec_batch = batch["target_batch"]
        dec_ext_batch = batch["target_ext_batch"]

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Embedding - context
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch) + emb_mask
        src_vad = enc_vad_batch  # (bsz, len, 1)

        if concept_input.size()[0] != 0:
            ## Embedding - concept
            mask_con = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
            con_mask = self.embedding(batch["mask_concept"])  # dialogue state
            con_emb = self.embedding(concept_input) + con_mask

            ## Knowledge Update
            src_emb = self.concept_graph(src_emb, con_emb,
                                         batch["adjacency_mask_batch"])  # (bsz, context+concept, emb_dim)
            mask_src = torch.cat((mask_src, mask_con), dim=2)  # (bsz, 1, context+concept)

            src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)  # (bsz, len)
        ## Encode - context & concept
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)

        ## emotional signal distilling
        if concept_input.size()[0] != 0:
            concept_context_vad = torch.cat((batch["context_vad"], batch["concept_vad_batch"]),
                                            dim=1)  # (bsz, context_len+concept_len)
        else:
            concept_context_vad = batch["context_vad"]
        concept_context_vad = torch.softmax(concept_context_vad, dim=-1).unsqueeze(2)
        concept_context_vad = concept_context_vad.repeat(1, 1, config.emb_dim)  # (bsz, len, emb_dim)
        emotion_context = torch.sum(concept_context_vad * encoder_outputs, dim=1)  # (bsz, emb_dim)
        emotion_contexts = concept_context_vad * encoder_outputs

        emotion_logit = self.identify(emotion_context)  # (bsz, emotion_num)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_logit, batch['emotion_label'])

        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["emotion_label"].cpu().numpy(), pred_emotion)

        # Decode
        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        dec_emb = self.embedding(dec_batch[:, :-1])  # (bsz, tgt_len, emb_dim)
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1)  # (bsz, tgt_len, emb_dim)

        mask_trg = dec_batch.data.eq(config.PAD_idx).unsqueeze(1)
        if "wo_E_CatM" in config.model:
            pre_logit, attn_dist = self.decoder(dec_emb, encoder_outputs, None, None, (mask_src, mask_trg))
        else:  # MK_EDG
            pre_logit, attn_dist = self.decoder(dec_emb, encoder_outputs, None, emotion_context, (mask_src, mask_trg))

        ## compute output dist
        if concept_input.size()[0] != 0:
            enc_ext_batch = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)
        else:
            enc_ext_batch = enc_batch_extend_vocab

        logit = self.generator(pre_logit, None, None, attn_dist, enc_ext_batch if config.pointer_gen else None,
                               max_oov_length, attn_dist_db=None)
        # logit = F.log_softmax(logit,dim=-1) #fix the name later
        ## loss: NNL if ptr else Cross entropy
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))

        loss += loss_emotion

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1)).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch_extend_vocab, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_vad_batch = batch['context_vad']
        enc_batch_extend_vocab = batch["context_ext_batch"]

        concept_input = batch["concept_batch"]  # (bsz, max_concept_len)
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        ## Encode - context
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch) + emb_mask
        src_vad = enc_vad_batch  # (bsz, len, 1)

        if concept_input.size()[0] != 0:
            mask_con = concept_input.data.eq(config.PAD_idx).unsqueeze(1)  # real mask
            con_mask = self.embedding(batch["mask_concept"])  # dialogue state
            con_emb = self.embedding(concept_input) + con_mask

            ## Knowledge Update
            src_emb = self.concept_graph(src_emb, con_emb,
                                         batch["adjacency_mask_batch"])  # (bsz, context+concept, emb_dim)
            mask_src = torch.cat((mask_src, mask_con), dim=2)  # (bsz, 1, context+concept)

            src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1)  # (bsz, len)
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)

        ## Identify
        if concept_input.size()[0] != 0:
            concept_context_vad = torch.cat((batch["context_vad"], batch["concept_vad_batch"]),
                                            dim=1)  # (bsz, context_len+concept_len)
        else:
            concept_context_vad = batch["context_vad"]
        concept_context_vad = torch.softmax(concept_context_vad, dim=-1).unsqueeze(2)
        concept_context_vad = concept_context_vad.repeat(1, 1, config.emb_dim)  # (bsz, len, emb_dim)
        emotion_context = torch.sum(concept_context_vad * encoder_outputs, dim=1)
        emotion_contexts = concept_context_vad * encoder_outputs

        emotion_logit = self.identify(emotion_context)  # (bsz, emotion_num)

        ## compute output dist
        if concept_input.size()[0] != 0:
            enc_ext_batch = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)
        else:
            enc_ext_batch = enc_batch_extend_vocab

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)  # (bsz, 1, emb_dim)
        sos_emb = ys_emb
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(encoder_outputs),
                                              (mask_src, mask_trg))
            else:
                out, attn_dist = self.decoder(ys_emb, encoder_outputs, None, emotion_context, (mask_src, mask_trg))

            prob = self.generator(out, None, None, attn_dist, enc_ext_batch, max_oov_length, attn_dist_db=None)
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent


