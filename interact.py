# Use this script to interact with the trained model.
import pdb
import torch
import torch.utils.data as data
from collections import deque
from utils import config
from utils.data_loader import prepare_data_seq
import json
import nltk
from utils.data_reader import Lang
from baselines.transformer import Transformer
from baselines.EmoPrepend import EmoP
from baselines.MoEL import MoEL
from Model.EmpDG_G import Empdg_G

EMODICT = json.load(open('empathetic-dialogue/NRCDict.json'))[0]
def get_emotion_words(utt_words):
    emo_ws = []
    for u in utt_words:
        for w in u.split():
            if w in EMODICT:
                emo_ws.append(w)
    return emo_ws

word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}

def clean(sentence, word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k,v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

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

        item["context_text"] = [x for x in self.data if x!="None"]
        item["emotion_context_text"] = get_emotion_words(item["context_text"])

        inputs = self.preprocess([item["context_text"],
                                  item["emotion_context_text"]])

        item["context"], item["context_ext"], item["oovs"], item["context_mask"], \
        item["emotion_context"], item["emotion_context_ext"], item["emotion_context_mask"] = inputs

        return item

    def __len__(self):
        return 1

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
            sentence = clean(sentence, word_pairs)
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
                sentence = clean(sentence, word_pairs)
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
    ## input - context
    context_batch = torch.LongTensor([batch_data[0]['context']])
    context_ext_batch = torch.LongTensor([batch_data[0]['context_ext']])
    mask_context = torch.LongTensor([batch_data[0]['context_mask']])  # (bsz, max_context_len) dialogue state

    ## input - emotion_context
    emotion_context_batch = torch.LongTensor([batch_data[0]['emotion_context']])
    emotion_context_ext_batch = torch.LongTensor([batch_data[0]['emotion_context_ext']])
    mask_emotion_context = torch.LongTensor([batch_data[0]['emotion_context_mask']])

    d = {}
    ##input
    d["context_batch"] = context_batch.to(config.device)  # (bsz, max_context_len)
    d["context_ext_batch"] = context_ext_batch.to(config.device)  # (bsz, max_context_len)
    d["mask_context"] = mask_context.to(config.device)

    d["emotion_context_batch"] = emotion_context_batch.to(config.device)  # (bsz, max_emo_context_len)
    d["emotion_context_ext_batch"] = emotion_context_ext_batch.to(config.device)  # (bsz, max_emo_context_len)
    d["mask_emotion_context"] = mask_emotion_context.to(config.device)

    ##text
    d["context_txt"] = [batch_data[0]['context_text']]
    d["emotion_context_txt"] = [batch_data[0]['emotion_context_text']]
    d["oovs"] = [batch_data[0]["oovs"]]
    return d


def make_batch(inp,vacab):
    d = Dataset(inp,vacab)
    loader = torch.utils.data.DataLoader(dataset=d, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return iter(loader).next()

if __name__ == '__main__':
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)
    if config.model == "Transformer":
        model = Transformer(vocab, decoder_number=program_number)

    if (config.model == "EmoPrepend") or (config.model == "EmpDG_woG"):
        model = EmoP(vocab, decoder_number=program_number)

    if config.model == "MoEL":  # see source code at: https://github.com/HLTCHKUST/MoEL
        model = MoEL(vocab, decoder_number=program_number)

    if (config.model == "EmpDG_woD") or (config.model == "EmpDG"):  # train/test for EmpDG_woD; test for EmpDG
        model = Empdg_G(vocab, emotion_number=program_number)

    checkpoint = torch.load('result/' + config.model + '_best.tar', map_location=lambda storage, location: storage)
    if config.model == "EmpDG" or config.model == "EmpDG_woG":
        weights_best = checkpoint['models_g']
    else:
        weights_best = checkpoint['models']
    model.load_state_dict({name: weights_best[name] for name in weights_best})
    model.to(config.device)
    model.eval()

    print('Let\'s chat')
    DIALOG_SIZE = 5
    context = deque(DIALOG_SIZE * ['None'], maxlen=DIALOG_SIZE)

    try:
        while True:
            ipt = input(">> User: ")
            if (len(str(ipt).strip()) != 0):
                context.append(str(ipt).rstrip().lstrip())
                batch = make_batch(context, vocab)
                sent_g = model.decoder_greedy(batch, max_dec_step=30)
                print("{}: ".format(config.model), sent_g[0])
                context.append(sent_g[0])
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from chatting .')



