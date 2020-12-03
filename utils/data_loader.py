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
        item["emotion_context_text"] = self.data["emotion_context"][index]
        item["target_text"] = self.data["target"][index]
        item["target_emotion_text"] = self.data["target_emotion"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["feedback_text"] = self.data["feedback"][index]
        item["feedback_emotion_text"] = self.data["feedback_emotion"][index]
        item["situation_text"] = self.data["situation"][index]

        inputs = self.preprocess([item["context_text"], item["emotion_context_text"]])
        item["context"], item["context_ext"], item["oovs"], item["context_mask"], \
        item["emotion_context"], item["emotion_context_ext"], item["emotion_context_mask"] = inputs

        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["target_ext"] = self.target_oovs(item["target_text"], item["oovs"])
        assert item["target"].size() == item["target_ext"].size()

        item["feedback"] = self.preprocess(item["feedback_text"], anw=True)
        item["feedback_emotion"] = self.preprocess(item["feedback_emotion_text"], anw=True)

        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)  # one-hot and scalar label
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

    def process_oov(self, context, emotion_context):  # oov for input
        ids = []
        ids_e = []
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

        for ew in emotion_context:
            if ew in self.vocab.word2index:
                i = self.vocab.word2index[ew]
                ids_e.append(i)
            elif ew in oovs:
                oov_num = oovs.index(ew)
                ids_e.append(len(self.vocab.word2index) + oov_num)
            else:
                oovs.append(ew)
                oov_num = oovs.index(w)
                ids_e.append(len(self.vocab.word2index) + oov_num)
        return ids, ids_e, oovs

    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if anw:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
            return torch.LongTensor(sequence)
        else:
            context = arr[0]
            emotion_context = arr[1]

            X_dial = [config.CLS_idx]
            X_dial_ext = [config.CLS_idx]
            X_dial_mask = [config.CLS_idx]

            X_emotion = [config.LAB_idx]
            X_emotion_ext = [config.LAB_idx]
            X_emotion_mask = [config.LAB_idx]


            for i, sentence in enumerate(context):  # concat sentences in context
                X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_dial_mask += [spk for _ in range(len(sentence))]

            for i, ew in enumerate(emotion_context):
                X_emotion += [self.vocab.word2index[ew] if ew in self.vocab.word2index else config.UNK_idx]
                X_emotion_mask += [self.vocab.word2index["LAB"]]

            X_ext, X_e_ext, X_oovs = self.process_oov(context, emotion_context)
            X_dial_ext += X_ext
            X_emotion_ext += X_e_ext

            assert len(X_dial) == len(X_dial_mask) == len(X_dial_ext)
            assert len(X_emotion) == len(X_emotion_ext) == len(X_emotion_mask)

            return X_dial, X_dial_ext, X_oovs, X_dial_mask, X_emotion, X_emotion_ext, X_emotion_mask

    def preprocess_emo(self, emotion, emo_map):
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]  # one


def collate_fn(batch_data):
    def merge(sequences):  # len(sequences) = bsz
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths

    batch_data.sort(key=lambda x: len(x["context"]), reverse=True)
    item_info = {}
    for key in batch_data[0].keys():
        item_info[key] = [d[key] for d in batch_data]

    ## input - context
    context_batch, context_lengths = merge(item_info['context'])
    context_ext_batch, _ = merge(item_info['context_ext'])
    mask_context, _ = merge(item_info['context_mask'])  # (bsz, max_context_len) dialogue state

    ## input - emotion_context
    emotion_context_batch, emotion_context_lengths = merge(item_info['emotion_context'])
    emotion_context_ext_batch, _ = merge(item_info['emotion_context_ext'])
    mask_emotion_context, _ = merge(item_info['emotion_context_mask'])

    ## Target
    target_batch, target_lengths = merge(item_info['target'])
    target_ext_batch, _ = merge(item_info['target_ext'])

    feedback_batch, feedback_lengths = merge(item_info['feedback'])
    feedback_emotion_batch, feedback_emotion_lengths = merge(item_info['feedback_emotion'])

    d = {}
    ##input
    d["context_batch"] = context_batch.to(config.device)  # (bsz, max_context_len)
    d["context_ext_batch"] = context_ext_batch.to(config.device)  # (bsz, max_context_len)
    d["context_lengths"] = torch.LongTensor(context_lengths).to(config.device)  # (bsz, )
    d["mask_context"] = mask_context.to(config.device)

    d["emotion_context_batch"] = emotion_context_batch.to(config.device)  # (bsz, max_emo_context_len)
    d["emotion_context_ext_batch"] = emotion_context_ext_batch.to(config.device)  # (bsz, max_emo_context_len)
    d["emotion_context_lengths"] = torch.LongTensor(emotion_context_lengths).to(config.device)  # (bsz, )
    d["mask_emotion_context"] = mask_emotion_context.to(config.device)

    ##output
    d["target_batch"] = target_batch.to(config.device)  # (bsz, max_target_len)
    d["target_ext_batch"] = target_ext_batch.to(config.device)
    d["target_lengths"] = torch.LongTensor(target_lengths).to(config.device)  # (bsz,)

    d["feedback_batch"] = feedback_batch.to(config.device)  # (bsz, max_target_len)
    d["feedback_lengths"] = torch.LongTensor(feedback_lengths).to(config.device)  # (bsz,)
    d["feedback_emotion_batch"] = feedback_emotion_batch.to(config.device)  # (bsz, max_target_len)

    ##program
    d["target_emotion"] = torch.LongTensor(item_info['emotion']).to(config.device)
    d["emotion_label"] = torch.LongTensor(item_info['emotion_label']).to(config.device)  # (bsz,)
    d["emotion_widx"] = torch.LongTensor(item_info['emotion_widx']).to(config.device)
    assert d["emotion_widx"].size() == d["emotion_label"].size()

    ##text
    d["context_txt"] = item_info['context_text']
    d["emotion_context_txt"] = item_info['emotion_context_text']
    d["target_txt"] = item_info['target_text']
    d["target_emotion_txt"] = item_info['target_emotion_text']
    d["feedback_txt"] = item_info['feedback_text']
    d["feedback_emotion_txt"] = item_info['feedback_emotion_text']
    d["situation_txt"] = item_info['situation_text']
    d["emotion_txt"] = item_info['emotion_text']
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


def prepare_data_seq(batch_size=32, adver_train=False, gen_data=False):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset(adver_train)  # read data from local
    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)

    dataset_test = Dataset(pairs_tst, vocab)
    if gen_data:
        bsz = batch_size
    else:
        bsz = 1
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=bsz,
                                                  shuffle=False, collate_fn=collate_fn)

    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)