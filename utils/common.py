from utils import config
import numpy as np
import pickle
import re
import json
import math
import pdb
import torch
import torch.nn as nn
from tqdm import tqdm
import nltk
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), config.PAD_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = config.SOS_idx

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == config.EOS_idx:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[config.SOS_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, model, lang):

        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words
        self.beam_size = config.beam_size
        self.device = torch.device('cuda' if config.USE_CUDA else 'cpu')

    def beam_search(self, batch, max_dec_step):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)

            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_encoder_db, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm,
                             enc_batch_extend_vocab, extra_zeros, mask_src, encoder_db, mask_transformer_db,
                             DB_ext_vocab_batch):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm, enc_batch_extend_vocab,
                             extra_zeros, mask_src, encoder_db, mask_transformer_db, DB_ext_vocab_batch):
                ## masking
                mask_trg = dec_seq.data.eq(config.PAD_idx).unsqueeze(1)
                mask_src = torch.cat([mask_src[0].unsqueeze(0)] * mask_trg.size(0), 0)
                dec_output, attn_dist = self.model.decoder(self.model.embedding(dec_seq), enc_output,
                                                           (mask_src, mask_trg))

                db_dist = None

                prob = self.model.generator(dec_output, attn_dist, enc_batch_extend_vocab, extra_zeros, 1, True,
                                            attn_dist_db=db_dist)
                # prob = F.log_softmax(prob,dim=-1) #fix the name later
                word_prob = prob[:, -1]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm, enc_batch_extend_vocab,
                                     extra_zeros, mask_src, encoder_db, mask_transformer_db, DB_ext_vocab_batch)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # -- Encode
            context_input = batch["context_batch"]  # (bsz, max_context_len)
            concept_input = batch["concept_batch"]  # (bsz, max_concept_len)

            ## Embedding
            semantic_embed = self.model.embedding(context_input)  # (bsz, max_context_len, emb_dim)
            concept_semantic_embed = self.model.embedding(concept_input)  # (bsz, max_concept_len, emb_dim)

            # Knowledge Update
            concept_context = self.model.concept_graph(semantic_embed, concept_semantic_embed,
                                                 batch["adjacency_mask_batch"])  # (bsz, context+concept, emb_dim)

            # Encode
            concept_context = concept_context.transpose(0, 1)
            concept_context_mask = torch.cat((batch["mask_context"], batch["mask_concept"]), dim=1)
            concept_context_mask = concept_context_mask.transpose(0, 1)
            context_resp = self.model.encoder(concept_context,
                                        concept_context_mask)  # (context_len+concept_len, bsz, emb_dim)

            # Identify
            ROOT_resp = context_resp[0, :, :]  # (bsz, emb_dim)
            emotion_logit = self.model.identify(ROOT_resp)  # (bsz, emotion_num)

            encoder_db = None
            mask_transformer_db = None
            DB_ext_vocab_batch = None

            # -- Repeat data for beam search
            n_bm = self.beam_size  # 5
            len_s, n_inst, d_h = context_resp.size()  # (src_len, bsz, emb_dim)
            # src_seq = enc_batch.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = context_resp.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)





            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # -- Decode
            for len_dec_seq in range(1, max_dec_step + 1):

                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc,
                                                        inst_idx_to_position_map, n_bm, None,
                                                        None, None, encoder_db, mask_transformer_db,
                                                        DB_ext_vocab_batch)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, encoder_db, src_enc, inst_idx_to_position_map = collate_active_info(src_seq, encoder_db,
                                                                                             src_enc,
                                                                                             inst_idx_to_position_map,
                                                                                             active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(' '.join([self.model.vocab.index2word[idx] for idx in d[0]]).replace('EOS', ''))

        return ret_sentences  # , batch_scores


def print_custum(emotion, dial, emotion_context, ref, hyp_g, hyp_b):
    print("emotion:{}".format(emotion))
    print("Context:{}".format(dial))
    print("Emotion_context:{}".format(emotion_context))
    print("Beam: {}".format(hyp_b))
    print("Greedy:{}".format(hyp_g))
    print("Ref:{}".format(ref))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand



def distinctEval(preds):
    response_ugm = set([])
    response_bgm = set([])
    response_len = sum([len(p) for p in preds])  

    for path in preds:
        for u in path:
            response_ugm.add(u)
        for b in list(nltk.bigrams(path)):  
            response_bgm.add(b)
    response_len_ave = response_len/len(preds)
    distinctOne = len(response_ugm)/response_len
    distinctTwo = len(response_bgm)/response_len
    return distinctOne, distinctTwo, response_len_ave


def evaluate(model, data, ty='valid', max_dec_step=30, adver_train=False):
    if adver_train is False:
        outputs = open("Predictions/{}.txt".format(config.model), "w", encoding="utf-8")
    model.eval()
    model.cuda()
    model.__id__logger = 0
    ref, hyp_g, hyp_b, hyp_t = [], [], [], []
    if ty == "test":
        print("testing generation:")
    l = []
    p = []
    bce = []
    acc = []

    res = {}
    itr = 0

    pbar = tqdm(enumerate(data), total=len(data))
    for j, batch in pbar:
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        if ty == "test" or adver_train:
            if adver_train:
                sent_g = model.predict(batch,max_dec_step=max_dec_step)  # sentences list, each sentence is a string.
            else:
                sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)  # sentences list, each sentence is a string.
            for i, greedy_sent in enumerate(sent_g):
                res[itr] = greedy_sent.split()
                itr += 1
                rf = " ".join(batch["target_txt"][i])
                hyp_g.append(greedy_sent)
                ref.append(rf)
                if adver_train is False:
                    print_custum(emotion=batch["emotion_txt"][i],
                                 dial=[" ".join(s) for s in batch['context_txt'][i]],
                                 emotion_context=str(batch['emotion_context_txt'][i]),
                                 ref=rf,
                                 hyp_g=greedy_sent,
                                 hyp_b=[])
                    outputs.write("emotion:{} \n".format(batch["emotion_txt"][i]))
                    outputs.write("Context:{} \n".format(
                        [" ".join(s) for s in batch['context_txt'][i]]))
                    outputs.write("Emotion_context:{} \n".format(batch["emotion_context_txt"][i]))
                    outputs.write("Feedback:{} \n".format(batch["feedback_txt"][i]))
                    outputs.write("Pred:{} \n".format(greedy_sent))
                    outputs.write("Ref:{} \n".format(rf))

        pbar.set_description("loss:{:.4f}; ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l))))
    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)


    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)
    print()
    print("EVAL\tLoss\tPPL\tAccuracy\tDist-1\tDist-2")
    print(
        "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}".format(ty, loss, math.exp(loss), acc,
                                                            mi_dist1 * 100, mi_dist2 * 100))
    if adver_train:
        print("yesss!!!!")
        return loss, math.exp(loss), bce, acc, mi_dist1 * 100, mi_dist2 * 100
    else:
        return loss, math.exp(loss), bce, acc

EMODICT = json.load(open('empathetic-dialogue/NRCDict.json'))[0]
def get_emotion_words(utt_words):
    emo_ws = []
    for u in utt_words:
        if u in EMODICT:
            emo_ws.append(u)
    return emo_ws

def gen_disc_train_data(model, data, ty='test', max_dec_step=50):
    output_data =  {'context':[],'emotion_context':[],'target':[],'target_emotion':[],'feedback':[],'feedback_emotion':[],'emotion':[],'situation':[],'pred':[],'pred_emotion':[]}
    model.__id__logger = 0
    print("---------------------------begin generating data for pre-train disc---------------------------")
    l = []
    p = []
    bce = []
    acc = []

    pbar = tqdm(enumerate(data), total=len(data))
    for j, batch in pbar:
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        sent_g = model.predict(batch, max_dec_step=max_dec_step)  # sentences list, each sentence is a string.
        for i, greedy_sent in enumerate(sent_g):
            output_data['target'].append(batch["target_txt"][i])
            output_data['target_emotion'].append(batch["target_txt"][i])
            output_data['context'].append(batch["context_txt"][i])
            output_data['emotion_context'].append(batch["emotion_context_txt"][i])
            output_data['emotion'].append(batch["emotion_txt"][i])
            output_data['feedback'].append(batch["feedback_txt"][i])
            output_data['feedback_emotion'].append(batch["feedback_emotion_txt"][i])
            output_data['situation'].append(batch["situation_txt"][i])

            if i == 0:
                print(greedy_sent.split())
                print(get_emotion_words(greedy_sent.split()))
                print("--------------")
            output_data['pred'].append(greedy_sent.split())
            output_data['pred_emotion'].append(get_emotion_words(greedy_sent.split()))
        pbar.set_description("loss:{:.4f}; ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l))))

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)
    print()
    print("EVAL\tLoss\tPPL\tAccuracy\n")
    print(
        "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ty, loss, math.exp(loss), acc))

    return output_data

def gen_disc_train_batch(model, data, ty='test', max_dec_step=50):
    output_data =  {'context':[],'emotion_context':[],'target':[],'target_emotion':[],'feedback':[],'feedback_emotion':[],'emotion':[],'situation':[],'pred':[],'pred_emotion':[]}
    model.__id__logger = 0
    l = []
    p = []
    bce = []
    acc = []

    for j, batch in enumerate(data):
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        sent_g = model.predict(batch, max_dec_step=max_dec_step)  # sentences list, each sentence is a string.
        for i, greedy_sent in enumerate(sent_g):
            output_data['target'].append(batch["target_txt"][i])
            output_data['target_emotion'].append(batch["target_txt"][i])
            output_data['context'].append(batch["context_txt"][i])
            output_data['emotion_context'].append(batch["emotion_context_txt"][i])
            output_data['emotion'].append(batch["emotion_txt"][i])
            output_data['feedback'].append(batch["feedback_txt"][i])
            output_data['feedback_emotion'].append(batch["feedback_emotion_txt"][i])
            output_data['situation'].append(batch["situation_txt"][i])

            if i == 0:
                print(greedy_sent.split())
                print(get_emotion_words(greedy_sent.split()))
                print("--------------")
            output_data['pred'].append(greedy_sent.split())
            output_data['pred_emotion'].append(get_emotion_words(greedy_sent.split()))

    return output_data

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def evaluate_disc(model, data, ty='test', max_dec_step=50):
    model.eval()
    model.cuda()
    model.__id__logger = 0
    l = []
    p = []
    bce = []
    acc = []

    res = {}
    itr = 0
    pbar = tqdm(enumerate(data), total=len(data))
    for j, batch in pbar:
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        sent_g = model.predict(batch, max_dec_step=max_dec_step)  # sentences list, each sentence is a string.
        for i, greedy_sent in enumerate(sent_g):
            res[itr] = greedy_sent.split()
            itr += 1

    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)
    print("Loss\tPPL\tAccuracy\tDist-1\tDist-2")
    print(
        "{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}".format(loss, math.exp(loss), acc, mi_dist1 * 100, mi_dist2 * 100))
    model.train()
    return loss, math.exp(loss), bce, acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def gleu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

class Embeddings(nn.Module):
    def __init__(self,vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

def share_embedding(vocab, pretrain=True):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    if pretrain:
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.lut.weight.data.requires_grad = True
    return embedding

def get_input_from_batch(batch):
    enc_batch = batch["input_batch"]
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size())

    if config.USE_CUDA:
        enc_padding_mask = enc_padding_mask.cuda()
        if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))












