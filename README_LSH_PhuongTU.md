# Phase 3 — 6-LSH: Lightweight Segmentation Head

**Người thực hiện: PhuongTU**

L0 là RepDWNet B1 + WDice với head cũ. L1 dùng logits decoder `/1 + /2`;
L2 dùng `/1 + /2 + /4`. Mỗi nhánh là Conv1×1 có bias, tạo 3 channel logits.
Resize logits về spatial size thật của feature /1 bằng bilinear,
`align_corners=False`, rồi trung bình đều trước một WDice duy nhất.
Encoder–decoder B1 vẫn train end-to-end, giữ resize nội bộ True. Public output
là tensor `[B,3,H,W]`. Không softmax/detach/auxiliary loss hoặc kết hợp SRR.

## Source và đối chứng

Remote: `https://github.com/FUAVX/RPD_Reproduce.git`; branch mới: `f/lsh`.
Nền đã đối chiếu là `7b19c752e146559c7ae81fd3abce3fd92bdda040` của `f/srr`:
B1/WDice/data/caller/config baseline ở đó trùng byte với nền WDice
`d77d518b8f650443216070fa76742db76e41b8ac`. Kế thừa source tooling/provenance;
hai model LSH kế thừa trực tiếp B1. Source SHA runtime được ghi đầy đủ trong
hai notebook: `e39b25018f27bf107a406d90270245ee80b0b233`; commit launcher được tạo sau source commit, không cần hash tự
tham chiếu. Các plan chuẩn và artifact bàn giao không bị sửa.

| Run | Model | Config | Params toàn model / head |
|---|---|---|---:|
| L0 | RepDWNet | `configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml` | 189827 / 51 |
| L1 | RepDWNetLSHL1 | `configs/p3_scenarios/P3_LSH_L1_WDice.yaml` | 189878 / 102 |
| L2 | RepDWNetLSHL2 | `configs/p3_scenarios/P3_LSH_L2_WDice.yaml` | 189977 / 201 |

Hai model `.py` nằm trong `models/rpdnet/`, factory có thêm hai lựa chọn.
Classifier `out_conv` /1 gốc vẫn được dùng; các head mới không để lại tham số
trainable thừa. Channel thực sau up4/up3/up2 là 16/16/32 ở base_c=16.

Giữ toàn bộ recipe ngoài experiment/backbone trong YAML mới: seed 1682409321;
100 epochs; LR 2e-4; cosine; warmup 5; weight_decay 2e-4; FP32; batch 4/GPU;
accumulate 1; crop train 768×768. **Optimizer thực là Adam** theo
`SegmentationNetwork.configure_optimizers`, không thay bằng AdamW.
WDice giữ weighted CE + macro Soft Dice, weights [1.47,5.06,10.02], lambda 1,
epsilon 1e-6, ignore 255 và masking/reduction hiện có. Validation không truyền
mask như train; preflight sẽ báo lỗi nếu nhãn val không phù hợp semantics đó.

## Checkpoint rule đã chọn trước train

So sánh L0/L1/L2 bằng **best `val_weed_iou`, mode=max**, dùng callback hiện có
`checkpoint_saver_val_weed_iou`; tên file chứa `best_weedIoU`. Giữ mặc định
save_top_k=1 và quy tắc hòa điểm của cùng phiên bản Lightning. Validation mỗi
5 epochs. `last.ckpt` dùng tiếp tục trạng thái, không làm tiêu chí thắng.
Các checkpoint best mIoU/loss và periodic vẫn được lưu bởi caller gốc; không
chọn riêng loại có lợi cho từng biến thể. Manifest của mọi run ghi cùng rule.

## Chạy trên Kaggle

1. Import [notebook L1](notebook/kaggle_train_p3_lsh_l1_wdice.ipynb) và
   [notebook L2](notebook/kaggle_train_p3_lsh_l2_wdice.ipynb) vào hai session
   GPU độc lập. Mỗi session mặc định dùng 2 GPU. Bật Internet.
2. Add Input PhenoBench có `train/images`, `train/semantics`, `val/images`,
   `val/semantics`, `test/images`. Không cần source ZIP. `DATASET_DIR` rỗng chỉ
   tự tìm root đã attach; không tải dataset. Nếu nhiều root, điền đường dẫn.
3. Chạy dependencies; restart kernel nếu được yêu cầu rồi chạy lại theo thứ tự.
   Giữ torch/torchvision/CUDA của Kaggle. Bộ pin kế thừa Lightning 1.9.5,
   torchmetrics 0.11.4, timm 0.9.16, numpy<2, setuptools<70; không khẳng định
   mọi package Kaggle tương thích với pin này.
4. Cell chuẩn bị clone `f/lsh`, checkout full SOURCE_COMMIT, tạo run token riêng
   và active config. Với 2 GPU, active config ghi devices=2, batch4/GPU và
   effective batch8. Không đổi số GPU/batch/precision riêng một run rồi coi
   tương đương đối chứng. Cần đổi thì áp dụng cùng điều kiện và ghi rõ.
5. Chạy logger smoke. Launcher sao `sitecustomize.py` ra ngoài source, đặt
   `P3_TENSORBOARD_NO_TF=1` và PYTHONPATH cho subprocess/DDP. TensorBoard dùng
   notf/tensorflow_stub, không kéo TensorFlow/JAX thật. Test ghi/đọc cả scalar,
   grouped scalar và Lightning logger; process con cũng xác nhận inheritance.
6. Chạy preflight: core checks normal import; sample train thật đã augment lặp
   batch4, forward/backward + Adam step; sample validation native lặp batch4;
   DDP hai bước synthetic với `find_unused_parameters=False` trên số GPU đã
   chọn. Mỗi notebook/model chạy subprocess riêng và không dùng report SRR
   hoặc report của model khác. Nếu chủ động dùng một GPU, DDP protocol test
   chuyển CPU và manifest ghi GPU-DDP chưa được thử.
7. Đọc actual shapes/peak memory rồi mới chạy cell train. GPU OOM hoặc label,
   logger, DDP lỗi đều chặn train; không tự giảm batch hay đổi loss. Loader gốc
   trả ảnh validation native, không thực thi center_crop 1024 trong YAML.
8. Tải artifacts ZIP và lưu output Kaggle. Có manifest, source SHA/diff/status,
   active config, runtime-fix hash/hook, versions, path/shape data, preflight,
   logs, checkpoints và digest checkpoint. ZIP không chứa source checkout.
   Có cell đóng gói lại sau khi ngắt có kiểm soát. Runtime bị kill hoàn toàn
   có thể không chạy finally; không nhận rằng checkpoint/artifacts luôn đủ.

Để chạy L0 với cùng launcher/guard và cùng source SHA, dùng một bản sao
notebook và đổi đúng hai giá trị trong cell chuẩn bị:

```python
SCENARIO = "P2_Ex7_RepDWNet_WDice-Loss"
CONFIG_RELATIVE = "configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml"
```

Giữ cùng GPU, dataset, recipe, startup hook, checkpoint rule và điều kiện đo.
Không dùng run B1 WCE cũ làm đối chứng duy nhất.

## Fresh và resume

Mặc định ba trường RESUME rỗng. Chỉ resume khi chỉ định checkpoint + saved
active_config.yaml + run_manifest.json từ cùng run. Không quét mọi last.ckpt.
Guard kiểm tra scenario/source, hash active config, runtime hook, checkpoint
và output bundle; config phải trùng chính xác, gồm data path/devices/workers.
Nếu chưa có digest checkpoint vì runtime bị kill trước packaging, phải giữ
checkpoint dưới `training_output/checkpoints/` trong bundle gốc bên cạnh
manifest/config; lúc chuẩn bị sẽ tính và lưu digest. Quan hệ bundle là yêu
cầu provenance của người chạy, không phải chứng nhận nguồn gốc nội tại của
file checkpoint. Chỉ dùng checkpoint đáng tin cậy: resume bật tải full-state
cho Lightning 1.9 trên PyTorch mới qua TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1.

Đổi source/config/hook/versions sau setup cần chuẩn bị run mới và preflight
lại. Các phép thử ở subprocess riêng không làm thay đổi RNG/BN của process
training mới. Run token/output/log/checkpoint tách biệt cho hai scenario.

## Kiểm tra và giới hạn

Bằng chứng và lệnh tái kiểm tra: [docs/lsh/VERIFICATION.md](docs/lsh/VERIFICATION.md).
CPU đã kiểm tra input chẵn/lẻ, công thức logits, shared weights/features khớp
L0, WDice/backward finite và gradient mọi tham số; head/total params đúng plan;
DDP hai process × hai bước cho từng model; logger event IO và inheritance.
PNG fixture kiểm tra remap raw3→1/4→2, crop train768 và validation native65×79.
Đó là fixture, chưa phải PhenoBench thật. Deploy random probe chỉ là diagnostic
chuẩn hóa theo scale output, không chứng nhận absolute tolerance trên model
đã train. Kết quả loss random không phải chất lượng segmentation.

Local không có CUDA hoặc PhenoBench thật; chưa kiểm tra GPU/full batch memory,
NCCL, checkpoint resume thực hay Kaggle sau sửa TensorBoard. Các kiểm tra còn
thiếu được đặt trước cell train. Chưa chạy training dài, chưa có kết luận
cải thiện weed/latency/memory. Weed IoU/mIoU thuộc training loop gốc; coverage
vùng nhỏ và miss rate cần evaluate riêng. Ngưỡng vùng nhỏ, seed bổ sung và
ngân sách latency/memory trong plan vẫn chưa được chốt bằng dữ liệu.
