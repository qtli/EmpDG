import json
from collections import defaultdict

def combine():
    transformer = open("Transformer.txt", "r", encoding="utf-8")
    emoprepend = open("EmoPrepend.txt", "r", encoding="utf-8")
    MoEL = open("MoEL.txt", "r", encoding="utf-8")  # Predicted by the author's released code: https://github.com/HLTCHKUST/MoEL
    wo_D = open("s_enc.txt", "r", encoding="utf-8")  #
    # wo_CE = open("wo_CE.txt", "r", encoding="utf-8")
    # Know_EDG = open("Know_EDG.txt", "r", encoding="utf-8")


    results_json = open("models_result.json", "w", encoding="utf-8")
    model_preds = defaultdict(list)

    for i, line in enumerate(transformer.readlines()):
        if line.startswith('Pred:'):
            pred = line.strip('Pred:').strip('\n')
            model_preds['transformer'].append(pred)

        if line.startswith('Ref:'):
            ref = line.strip('Ref:').strip('\n')
            model_preds['ref'].append(ref)

    for i, line in enumerate(emoprepend.readlines()):
        if line.startswith('Pred:'):
            pred = line.strip('Pred:').strip('\n')
            model_preds['emoprepend'].append(pred)


    for j, jline in enumerate(MoEL.readlines()):
        if jline.startswith('Greedy:'):
            exp = jline.strip('Greedy:').strip('\n')
            model_preds['MoEL'].append(exp)


    for i, line in enumerate(wo_D.readlines()):
        if line.startswith('Pred:'):
            pred = line.strip('Pred:').strip('\n')
            model_preds['wo_D'].append(pred)


    # for i, line in enumerate(wo_CE.readlines()):
    #     if line.startswith('Pred:'):
    #         pred = line.strip('Pred:').strip('\n')
    #         model_preds['wo_CE'].append(pred)
    #
    # for i, line in enumerate(Know_EDG.readlines()):
    #     if line.startswith('Pred:'):
    #         pred = line.strip('Pred:').strip('\n')
    #         model_preds['Know_EDG'].append(pred)
    json.dump(model_preds, results_json)


combine()

gdn = {}
predt = {}
prede = {}
predm = {}
pred_wo_d = {}
# pred_wo_g = {}
pred_empdg = {}

pred = {}

f = json.load(open("models_result.json", "r", encoding="utf-8"))

golds = f['ref']
transformers = f['transformer']
emoprepend = f['emoprepend']
moel = f['MoEL']
wo_d = f['wo_D']
# wo_g = f['wo_G']
# empdg = f['Know_EDG']

for i, p in enumerate(transformers):
    gdn[i] = [golds[i].split()]
    predt[i] = p.split()
    prede[i] = emoprepend[i].split()
    predm[i] = moel[i].split()
    pred_wo_d[i] = wo_d[i].split()
    # pred_wo_g[i] = wo_g[i].split()
    # pred_empdg[i] = empdg[i].split()

pred["Transformer"] = predt
pred["EmotionPrepend"] = prede
pred["MoEL"] = predm
pred["wo_D"] = pred_wo_d
# pred["wo_G"] = pred_wo_g
# pred["EmpDG"] = pred_empdg



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


for m, res in pred.items():
    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)

    print(m)
    print("mi_dist1", mi_dist1)
    print("mi_dist2", mi_dist2)
    print("avg_len", avg_len)

    print("& %.2f & %.2f & %.2f" \
          % (mi_dist1 * 100, mi_dist2 * 100, avg_len))
    print()





