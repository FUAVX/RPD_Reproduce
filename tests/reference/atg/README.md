# Official Attention U-Net reference

Source: https://github.com/ozan-oktay/Attention-Gated-Networks/tree/eee4881fdc31920efd873773e0b744df8dacbfb6

Files retain original author's code for source inspection and independent gate parity tests. Do not import the complete legacy repository at runtime; no medical dataset dependencies are needed. Tests execute only the original Generic2D gate and initialization definitions.

Adapters: explicit factor(2,2), bilinear align_corners=False, modern PyTorch API spelling. GenCom adapts CT-Single Conv3D/BN3D combine to Conv2D/BN2D. PRG/FBG bind to P2 features; no DSV head, medical loss or author optimizer is copied. Conv biases remain constructor defaults; no pass-through bias is added.
