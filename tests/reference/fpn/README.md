# Official reference (inspection fixture only)

Source: https://github.com/facebookresearch/Detectron/tree/04155a01a6ea68f22ac27c79a822066457941ece

`FPN.py` is the original FAIR/Caffe2 implementation, kept with its license for source inspection. Do not import it in the training runtime; Caffe2/Detectron is not a dependency.

Host adaptations: bilinear align_corners=True; width32; P5=M5 without output3x3; DW/PW option; final Mean and single segmentation head. Level names follow x2..x5 (strides2..16), not Detectron's strides4..32.

Initialization follows official fvcore `c2_xavier_fill`: kaiming_uniform_(a=1), bias0. Extension to DW/PW, A lateral and segmentation head is an approved team policy. Control/RepDW initialization remains unchanged.
