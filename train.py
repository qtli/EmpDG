import os
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.nn.init import xavier_uniform_

from utils.data_loader import prepare_data_seq
from utils.common import *

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
from utils.data_reader import Lang
from baselines.transformer import Transformer
from baselines.EmoPrepend import EmoP
from baselines.MoEL import MoEL
from Model.Empdg_G import Empdg_G


os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))

if __name__ == '__main__':
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

    if config.model == "Transformer":
        model = Transformer(vocab, decoder_number=program_number)

    if config.model == "EmoPrepend":
        model = EmoP(vocab, decoder_number=program_number)

    if config.model == "MoEL":  # see source code at: https://github.com/HLTCHKUST/MoEL
        model = MoEL(vocab, decoder_number=program_number)

    if (config.model == "EmpDG_woD") or (config.model == "EmpDG"):  # train/test or test
        model = Empdg_G(vocab, emotion_number=program_number)

    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("MODEL USED", config.model)
    print("TRAINABLE PARAMETERS", count_parameters(model))

    if config.test is False:
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
                loss, ppl, bce, acc = model.train_one_batch(next(data_iter),n_iter)
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

        ## TESTING
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        model.epoch = 100
        loss_test, ppl_test, bce_test, acc_test = evaluate(model, data_loader_tst, ty="test", max_dec_step=50)
    else:  # test
        print("TESTING !!!")
        model.cuda()
        model = model.eval()
        if config.specify_model:
            checkpoint = torch.load(config.resume_path)
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load('result/' + config.model + '_best.tar', map_location=lambda storage, location: storage)
            weights_best = checkpoint['models']
            model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        loss_test, ppl_test, bce_test, acc_test = evaluate(model, data_loader_tst, ty="test", max_dec_step=50)
    print("Model: ", config.model, "End .")
    if config.specify_model:
        file_summary = "_summary.txt"
    else:
        file_summary = config.save_path + "summary.txt"
    with open(file_summary, 'w') as the_file:
        the_file.write("EVAL\tLoss\tPPL\tAccuracy\n")
        the_file.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}".format("test", loss_test, ppl_test, acc_test))

