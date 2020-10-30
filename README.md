# EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation (COLING 2020)

This repository is the implementation of our COLING 2020 Paper [EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation](https://github.com/qtli/EmpDG).
This work was partly supported by the Tencent AI Lab Rhino-Bird Focused Research Program (JR201932).


The details of code and data will be updated before the end of October. Stay tuned!

## Model Architecture

![Image of EmpDG](fig/empdg.jpg)

## Experiment
***Quick Result***

To skip training, please check ***generation_result.txt***.

***Dependency***

Check the packages needed or simply run the command:
```console
❱❱❱ pip install -r requirements.txt
```
[**Pre-trained glove embedding**](http://nlp.stanford.edu/data/glove.6B.zip): ***glove.6B.300d.txt***, stored in `/vectors/`.


***Dataset***

The dataset (empathetic-dialogue) is preprocessed and stored in pickle format: 
```
.
└── empathetic-dialogue
    └── my_D_dataset_preproc.p
    └── my_G_dataset_preproc.p
```

***Pre-training Empathetic Generator (EmpDG_woD)***
```console
❱❱❱ python3 train.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --cuda --pretrain_emb --model wo_D --device_id 0 --save_path save/wo_D/ --pointer_gen
```

***EmpDG_woG: EmpDG not considering Multi-resolution Emotion Perception component***
```console
❱❱❱ python3 train.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --cuda --pretrain_emb --model wo_G --device_id 0 --save_path save/wo_G/ --pointer_gen
```

***EmpDG***
```console
❱❱❱ python3 train.py --cuda --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --cuda --pretrain_emb --model EmpDG --device_id 0 --save_path save/EmpDG/ --pointer_gen
```





## Reference & Acknowledgements


If you find our work useful, please cite our paper as follows:

```bibtex
@inproceedings{li-etal-2020-empdg,
  title={EmpDG: Multi-resolution Interactive Empathetic Dialogue Generation},
  author={Qintong Li and Hongshen Chen and Zhaochun Ren and Zhaopeng Tu and Zhumin Chen},
  booktitle={COLING},
  year={2020},
}
```







