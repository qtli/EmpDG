
import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import pprint
import pdb
pp = pprint.PrettyPrinter(indent=1)
import re
import ast
#from utils.nlp import normalize
import time
from collections import defaultdict
from utils.data_reader import load_dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}
        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        # item["target_vads_text"] = self.data["target_vads"][index]
        # item["target_vad_text"] = self.data["target_vad"][index]

        # process input utterances, concepts, vad, vads
        inputs = self.preprocess([item["context_text"],
                                  self.data["vads"][index],
                                  self.data["vad"][index],
                                  self.data["concepts"][index]])
        item["context"], item["context_ext"], item["oovs"], item["context_mask"], item["vads"], item["vad"], \
        item["concept_text"], item["concept"], item["concept_ext"], item["concept_vads"], item["concept_vad"] = inputs

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_ext"] = self.target_oovs(item["target_text"], item["oovs"])
        assert item["target"].size() == item["target_ext"].size()
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)  # one-hot and scalor label
        item["emotion_widx"] = self.vocab.word2index[item["emotion_text"]]

        return item

    def __len__(self):
        return len(self.data["target"])

    def target_oovs(self, target, oovs):  #
        ids = []  
        for w in target:
            if w not in self.vocab.word2index:
                if w in oovs:
                    ids.append(len(self.vocab.word2index) + oovs.index(w))
                else:
                    ids.append(config.UNK_idx)
            else:
                ids.append(self.vocab.word2index[w])
        ids.append(config.EOS_idx)
        return torch.LongTensor(ids)

    def process_oov(self, context, concept):  #
        ids = [] 
        oovs = []
        for si, sentence in enumerate(context):
            for w in sentence:
                if w in self.vocab.word2index:
                    i = self.vocab.word2index[w]
                    ids.append(i)
                else:  
                    if w not in oovs:  
                        oovs.append(w)
                    oov_num = oovs.index(w)  
                    ids.append(len(self.vocab.word2index) + oov_num)

        for sentence_concept in concept:
            for token_concept in sentence_concept:
                for c in token_concept:
                    if c not in oovs and c not in self.vocab.word2index:
                        oovs.append(c)
        return ids, oovs

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if anw:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            context = arr[0]
            context_vads = arr[1]
            context_vad = arr[2]
            concept = [arr[3][l][0] for l in range(len(arr[3]))]
            concept_vads = [arr[3][l][1] for l in range(len(arr[3]))]
            concept_vad = [arr[3][l][2] for l in range(len(arr[3]))]


            X_dial = [config.CLS_idx] 
            X_dial_ext = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            X_vads = [[0.5, 0.0, 0.5]]
            X_vad = [0.5]

            X_concept_text = defaultdict(list)  
            X_concept = [[]]  
            X_concept_ext = [[]]  
            X_concept_vads = [[0.5, 0.0, 0.5]]  
            X_concept_vad = [0.5] 
            assert len(context) == len(concept)

            X_ext, X_oovs = self.process_oov(context, concept)
            X_dial_ext += X_ext

            for i, sentence in enumerate(context):  
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]
                X_vads += context_vads[i]
                X_vad += context_vad[i]

                for j, token_conlist in enumerate(concept[i]):  
                    if token_conlist == []:
                        X_concept.append([])  
                        X_concept_ext.append([]) 
                        X_concept_vads.append([0.5, 0.0, 0.5])
                        X_concept_vad.append(0.5)
                    else:
                        X_concept_text[sentence[j]] += token_conlist[:config.concept_num]
                        X_concept.append([self.vocab.word2index[con_word] if con_word in self.vocab.word2index else config.UNK_idx for con_word in token_conlist[:config.concept_num]])

                        con_ext = []
                        for con_word in token_conlist[:config.concept_num]:
                            if con_word in self.vocab.word2index:
                                con_ext.append(self.vocab.word2index[con_word])
                            else:
                                if con_word in X_oovs:
                                    con_ext.append(X_oovs.index(con_word) + len(self.vocab.word2index))
                                else:
                                    con_ext.append(config.UNK_idx)
                        X_concept_ext.append(con_ext)
                        X_concept_vads.append(concept_vads[i][j][:config.concept_num])
                        X_concept_vad.append(concept_vad[i][j][:config.concept_num])

                        assert len([self.vocab.word2index[con_word] if con_word in self.vocab.word2index else config.UNK_idx for con_word in token_conlist[:config.concept_num]]) == len(concept_vads[i][j][:config.concept_num]) == len(concept_vad[i][j][:config.concept_num])
            assert len(X_dial) == len(X_mask) == len(X_concept) == len(X_concept_vad) == len(X_concept_vads)

            return X_dial, X_dial_ext, X_oovs, X_mask, X_vads, X_vad, X_concept_text, X_concept, X_concept_ext, X_concept_vads, X_concept_vad

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]  # one


def collate_fn(batch_data):
    '''
    :param batch_data: dict
    :return:
    '''
    def merge(sequences):  # len(sequences) = bsz
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long() ## padding index 1 1=True, in mask means padding.
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    def merge_concept(samples, samples_ext, samples_vads, samples_vad):
        concept_lengths = []
        token_concept_lengths = []
        concepts_list = []
        concepts_ext_list = []
        concepts_vads_list = []
        concepts_vad_list = []

        for i, sample in enumerate(samples):
            length = 0  
            sample_concepts = [] 
            sample_concepts_ext = []
            token_length = []
            vads = []
            vad = []

            for c, token in enumerate(sample):
                if token == []:  
                    token_length.append(0) 
                    continue
                length += len(token) 
                token_length.append(len(token)) 
                sample_concepts += token  
                sample_concepts_ext += samples_ext[i][c]
                vads += samples_vads[i][c]
                vad += samples_vad[i][c]
            assert len(token_length) == len(sample), "length is inconsistent"

            if length > config.total_concept_num:
                value, rank = torch.topk(torch.LongTensor(vad), k=config.total_concept_num)

                new_length = 0
                new_sample_concepts = []  
                new_sample_concepts_ext = []  
                new_token_length = []  
                new_vads = []
                new_vad = []

                cur_idx = 0
                for ti, token in enumerate(sample):
                    if token == []: 
                        new_token_length.append(0)
                        continue
                    top_length = 0  
                    for ci, con in enumerate(token):
                        point_idx = cur_idx + ci
                        if point_idx in rank: 
                            top_length += 1
                            new_length += 1
                            new_sample_concepts.append(con)
                            new_sample_concepts_ext.append(samples_ext[i][ti][ci])
                            new_vads.append(samples_vads[i][ti][ci])
                            new_vad.append(samples_vad[i][ti][ci])
                            assert len(samples_vads[i][ti][ci]) == 3

                    new_token_length.append(top_length)
                    cur_idx += len(token)

                concept_lengths.append(new_length) 
                token_concept_lengths.append(new_token_length)
                concepts_list.append(new_sample_concepts)
                concepts_ext_list.append(new_sample_concepts_ext)
                concepts_vads_list.append(new_vads)
                concepts_vad_list.append(new_vad)
                assert len(new_sample_concepts) == len(new_vads) == len(new_vad) == len(new_sample_concepts_ext), "concept token 与 vads [*,*,*] 数目 和 vad * 数目是对应的"
                assert len(new_token_length) == len(token_length)
            else:
                concept_lengths.append(length)  
                token_concept_lengths.append(token_length)
                concepts_list.append(sample_concepts)
                concepts_ext_list.append(sample_concepts_ext)
                concepts_vads_list.append(vads)
                concepts_vad_list.append(vad)

        if max(concept_lengths) != 0:
            padded_concepts = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len)
            padded_concepts_ext = torch.ones(len(samples), max(concept_lengths)).long() ## padding index 1 (bsz, max_concept_len)
            padded_concepts_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(samples), max(concept_lengths), 1) ## padding index 1 (bsz, max_concept_len)
            padded_concepts_vad = torch.FloatTensor([[0.5]]).repeat(len(samples), max(concept_lengths))  ## padding index 1 (bsz, max_concept_len)
            padded_mask = torch.ones(len(samples), max(concept_lengths)).long()  # Dialogue state

            for j, concepts in enumerate(concepts_list):
                end = concept_lengths[j]
                if end == 0:
                    continue
                padded_concepts[j, :end] = torch.LongTensor(concepts[:end])
                padded_concepts_ext[j, :end] = torch.LongTensor(concepts_ext_list[j][:end])
                padded_concepts_vads[j, :end, :] = torch.FloatTensor(concepts_vads_list[j][:end])
                padded_concepts_vad[j, :end] = torch.FloatTensor(concepts_vad_list[j][:end])
                padded_mask[j, :end] = config.ROOT_idx  # for DIALOGUE STATE

            return padded_concepts, padded_concepts_ext, concept_lengths, padded_mask, token_concept_lengths, padded_concepts_vads, padded_concepts_vad
        else:
            return torch.Tensor([]), torch.LongTensor([]), torch.LongTensor([]), torch.BoolTensor([]), torch.LongTensor([]), torch.Tensor([]), torch.Tensor([])

    def merge_vad(vads_sequences, vad_sequences):  # for context
        lengths = [len(seq) for seq in vad_sequences]  
        padding_vads = torch.FloatTensor([[[0.5, 0.0, 0.5]]]).repeat(len(vads_sequences), max(lengths), 1)
        padding_vad = torch.FloatTensor([[0.5]]).repeat(len(vads_sequences), max(lengths))

        for i, vads in enumerate(vads_sequences):
            end = lengths[i]  # context长度
            padding_vads[i, :end, :] = torch.FloatTensor(vads[:end])
            padding_vad[i, :end] = torch.FloatTensor(vad_sequences[i][:end])
        return padding_vads, padding_vad  # (bsz, max_context_len, 3); (bsz, max_context_len)

    def adj_mask(context, context_lengths, concepts, token_concept_lengths):
        '''

        :param self:
        :param context: (bsz, max_context_len)
        :param context_lengths: [] len=bsz
        :param concepts: (bsz, max_concept_len)
        :param token_concept_lengths: [] len=bsz; 
        :return:
        '''
        bsz, max_context_len = context.size()
        max_concept_len = concepts.size(1)  
        adjacency_size = max_context_len + max_concept_len
        adjacency = torch.ones(bsz, max_context_len, adjacency_size)   ## padding index 1 1=True

        for i in range(bsz):
            # ROOT -> TOKEN
            adjacency[i, 0, :context_lengths[i]] = 0  
            adjacency[i, :context_lengths[i], 0] = 0 

            con_idx = max_context_len    
            for j in range(context_lengths[i]):
                adjacency[i, j, j - 1] = 0 

                token_concepts_length = token_concept_lengths[i][j]  
                if token_concepts_length == 0:
                    continue
                else:
                    adjacency[i, j, con_idx:con_idx+token_concepts_length] = 0  
                    adjacency[i, 0, con_idx:con_idx+token_concepts_length] = 0 
                    con_idx += token_concepts_length 
        return adjacency

    batch_data.sort(key=lambda x: len(x["context"]), reverse=True) 
    item_info = {}
    for key in batch_data[0].keys(): 
        item_info[key] = [d[key] for d in batch_data]

    assert len(item_info['context']) == len(item_info['vad'])

    ## input - context
    context_batch, context_lengths = merge(item_info['context']) 
    context_ext_batch, _ = merge(item_info['context_ext'])
    mask_context, _ = merge(item_info['context_mask'])  # (bsz, max_context_len) dialogue state

    ## input - vad
    context_vads_batch, context_vad_batch = merge_vad(item_info['vads'], item_info['vad'])  # (bsz, max_context_len, 3); (bsz, max_context_len)

    assert context_batch.size(1) == context_vad_batch.size(1)


    ## input - concepts, vads, vad
    concept_batch, concept_ext_batch, concept_lengths, mask_concept, token_concept_lengths, concepts_vads_batch, concepts_vad_batch = \
        merge_concept(item_info['concept'], item_info['concept_ext'], item_info["concept_vads"], item_info["concept_vad"])  # (bsz, max_concept_len),

    ## input - adja_mask (bsz, max_context_len, max_context_len+max_concept_len)
    if concept_batch.size()[0] != 0:
        adjacency_mask_batch = adj_mask(context_batch, context_lengths, concept_batch, token_concept_lengths)
    else:
        adjacency_mask_batch = torch.Tensor([])


    ## Target
    target_batch, target_lengths = merge(item_info['target']) 
    target_ext_batch, _ = merge(item_info['target_ext'])

    d = {}
    ##input
    d["context_batch"] = context_batch.to(config.device)  # (bsz, max_context_len)
    d["context_ext_batch"] = context_ext_batch.to(config.device)  # (bsz, max_context_len)
    d["context_lengths"] = torch.LongTensor(context_lengths).to(config.device)  # (bsz, )
    d["mask_context"] = mask_context.to(config.device) 
    d["context_vads"] = context_vads_batch.to(config.device)   # (bsz, max_context_len, 3)
    d["context_vad"] = context_vad_batch.to(config.device)  # (bsz, max_context_len)

    ##concept
    d["concept_batch"] = concept_batch.to(config.device)  # (bsz, max_concept_len)
    d["concept_ext_batch"] = concept_ext_batch.to(config.device)  # (bsz, max_concept_len)
    d["concept_lengths"] = torch.LongTensor(concept_lengths).to(config.device)  # (bsz)
    d["mask_concept"] = mask_concept.to(config.device)  # (bsz, max_concept_len) 
    d["concept_vads_batch"] = concepts_vads_batch.to(config.device)  # (bsz, max_concept_len, 3)
    d["concept_vad_batch"] = concepts_vad_batch.to(config.device)   # (bsz, max_concept_len)
    d["adjacency_mask_batch"] = adjacency_mask_batch.bool().to(config.device)

    ##output
    d["target_batch"] = target_batch.to(config.device)  # (bsz, max_target_len)
    d["target_ext_batch"] = target_ext_batch.to(config.device)
    d["target_lengths"] = torch.LongTensor(target_lengths).to(config.device)  # (bsz,)

    ##program
    d["target_emotion"] = torch.LongTensor(item_info['emotion']).to(config.device)  
    d["emotion_label"] = torch.LongTensor(item_info['emotion_label']).to(config.device)  # (bsz,)
    d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx']).to(config.device)
    assert d["emotion_widx"].size() == d["emotion_label"].size()

    ##text
    d["context_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["emotion_txt"] = item_info['emotion_text']
    d["concept_txt"] = item_info['concept_text']
    d["oovs"] = item_info["oovs"]

    return d


def write_config():
    if not config.test:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(config.save_path+'config.txt', 'w') as the_file:
            for k, v in config.arg.__dict__.items():
                if "False" in str(v):
                    pass
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k,v))


def prepare_data_seq(batch_size=32):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()  # read data

    logging.info("Vocab  {} ".format(vocab.n_words))

    # change
    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                 batch_size=batch_size,
                                                 shuffle=True, collate_fn=collate_fn)
    #print('val len:',len(dataset_valid))
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                 batch_size=1,
                                                 shuffle=False, collate_fn=collate_fn)
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)