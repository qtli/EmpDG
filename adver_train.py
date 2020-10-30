import json
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.nn.init import xavier_uniform_

from utils.data_loader import prepare_data_seq
from utils.common import *
from train import *
from tensorboardX import SummaryWriter
import utils.config as config


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
from utils.data_reader import Lang
from baselines.transformer import Transformer
from baselines.EmoPrepend import EmoP

from Model.EmpDG import EmpDG
from Model.EmpDG_woD import EmpDG_woD
# from Model.EmpDG_woG import EmpDG_woG
from Model.EmpDG_D import EmpDG_D

os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))

def train_g(model):
    config.model = "wo_D"  # read training data for g
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    check_iter = 2000
    try:
        if config.USE_CUDA:
            model.cuda()
        model = model.train()
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(data_loader_tra)

        for n_iter in tqdm(range(1000000)):
            loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_train': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
            if config.noam:
                writer.add_scalars('lr', {'learning_rate': model.optimizer._rate}, n_iter)

            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                model.epoch = n_iter
                model.__id__logger = 0
                loss_val, ppl_val, bce_val, acc_val = evaluate(model, data_loader_val, ty="valid", max_dec_step=50)
                writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
                model = model.train()

                if n_iter < 13000:
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    ## SAVE MODEL
                    model_save_path = os.path.join(config.save_path,
                                                   'model_{}_{:.4f}'.format(iter, best_ppl))
                    torch.save(model.state_dict(), model_save_path)
                    weights_best = deepcopy(model.state_dict())
                    print("best_ppl: {}; patient: {}".format(best_ppl, patient))
                else:
                    patient += 1
                if patient > 2: break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    ## SAVE THE BEST
    torch.save({"models": weights_best,
                'result': [loss_val, ppl_val, bce_val, acc_val], },
               os.path.join('result/' + config.model + '_best.tar'))
    return model


def pre_train_g(model, resume=True):
    model.eval()
    if resume:
        checkpoint = torch.load('result/wo_D_best.tar', map_location=lambda storage, location: storage)
        weights_best = checkpoint['models']
        model.load_state_dict({name: weights_best[name] for name in weights_best})
    else:
        model = train_g(model)
        model.eval()
    return model



def gen_disc_data(model_g, epoch=0):
    # load data and generate predictions using model_g.
    config.model = "EmpDG"
    config.adver_train = True
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, adver_train=True)

    model_g.cuda()
    model_g.eval()

    output_train = gen_disc_train_data(model_g, data_loader_tra)  # obtain predicted response and its emotional words.
    print("complete training data.")
    output_dev = gen_disc_train_data(model_g, data_loader_val)
    print("complete dev data.")
    output_test = gen_disc_train_data(model_g, data_loader_val)
    print("complete test data.")

    # save data
    with open("empathetic-dialogue/adver_train/disc_data.p", "wb") as f:
        pickle.dump([output_train, output_dev, output_test], f)
        f.close()


def g_for_d(model_g, batch):
    enc_batch = batch["context_batch"]
    enc_emo_batch = batch['emotion_context_batch']

    ## Semantic Understanding
    mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
    sem_emb_mask = model_g.embedding(batch["mask_context"])  # dialogue state  E_d
    sem_emb = model_g.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
    sem_encoder_outputs = model_g.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

    ## Multi-resolution Emotion Perception (understanding & predicting)
    # mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
    # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
    # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
    # emo_encoder_outputs = self.emotion_pec(emo_emb, mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)
    mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
    # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
    # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
    emo_encoder_outputs = model_g.emotion_pec(model_g.embedding(enc_emo_batch),
                                           mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

    return sem_encoder_outputs[:, 0, :], emo_encoder_outputs[:, 0, :]


def pre_train_d(model_g, model_d, epoch=2):
    # prepare dataset
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, adver_train=True)

    if config.USE_CUDA:
        model_d.cuda()
    model_d = model_d.train()
    model_g = model_g.eval()
    writer = SummaryWriter(log_dir="save/pre_train_d/")

    weights_best = deepcopy(model_d.state_dict())
    data_iter = make_infinite(data_loader_tra)

    for n_iter in tqdm(range(2000)):
        # using model_g get context AND emotion context
        context, emotion_context = g_for_d(model_g, next(data_iter))

        # train semantic_d
        loss_d, loss_g = model_d.train_one_batch(context, emotion_context, next(data_iter))
        writer.add_scalars('loss_d', {'loss_d': loss_d}, n_iter)
        writer.add_scalars('loss_g', {'loss_g': loss_g}, n_iter)
        if config.noam:
            writer.add_scalars('lr', {'learning_rate': model_d.optimizer._rate}, n_iter)

    model_save_path = os.path.join('save/pre_train_d/model_d_pre_train')
    torch.save(model_d.state_dict(), model_save_path)




if __name__ == '__main__':
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

    if "wo_G" in config.model:
        model = EmpDG_woG(vocab, emotion_number=program_number)
    if config.model == "EmpDG":
        print('STEP 1: Pre-train Empathetic Generator ...')
        model_g = EmpDG_woD(vocab, emotion_number=program_number)
        model_g = adver_train.pre_train_g(model_g, resume=config.resume_g)

        print('STEP 2: Generating training data for two discriminators ...')
        adver_train.gen_disc_data(model_g)

        print('STEP 3: Pre-train Discriminators...')
        model_d = EmpDG_D()
        model_d = adver_train.pre_train_d(model_g, model_d, epoch=2)

        print("STEP 4: Adversarial joint learning")
        adver_train.adver_joint_train_gd(model_g, model_d)

        model = EmpDG(vocab, emotion_number=program_number, model_g=model_g)










