# Official later PSPNet source snapshot

Repository: https://github.com/hszhao/semseg
Commit: 4f274c3f276778228bc14a4565822d46359f0cc8

Unmodified references: model/pspnet.py -> pspnet.py; tool/train.py -> train.py.
Upstream LICENSE retained. This is the later PyTorch author implementation,
not a claim to reproduce the original CVPR2017 Caffe experiment.

Team adaptations: RepDWNet x3..x7, C/2 hidden width, production WDice,
P2 Adam LR/WD (no author 10x LR groups), per-GPU BN, from-scratch P2 recipe.
