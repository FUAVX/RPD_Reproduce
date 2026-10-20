# Phase 3 SRR — PhuongTU, GitHub workflow

Hai biến thể kế thừa source đã triển khai trong bộ bàn giao ngày 20/10/2026,
được tích hợp chọn lọc trên `f/wdice-loss` tại
`d77d518b8f650443216070fa76742db76e41b8ac`. Branch làm việc: `codex/p3-srr`.
Không merge `main`; checkout `p2_ende` ban đầu được giữ nguyên trong worktree cũ.

## Source và launcher

- PA1: [RepDWNetH8](models/rpdnet/RepDWNetH8.py) giữ block sâu ở /8,
  bỏ pool cuối và upsample đầu; giữ cùng tên/tham số với B1.
- PA2: [RepDWNetH4H8Fusion](models/rpdnet/RepDWNetH4H8Fusion.py) xử lý nhánh
  /4 (32 channel) và /8 (128 channel), hai exchange hai chiều đồng thời từ
  `high_pre`/`low_pre`, cập nhật skip /4; không có /16 hoặc tham số `down4` thừa.
  `final_only` là tùy chọn C0, không phải YAML thí nghiệm chính.
- Factory chỉ được bổ sung hai nhánh model. B1, RepDW/RPD blocks, WDice,
  dataset, training loop và YAML P2 được giữ nguyên.

Notebook PA1: [kaggle_train_p3_srr_h8_wdice.ipynb](notebook/kaggle_train_p3_srr_h8_wdice.ipynb).
Notebook PA2: [kaggle_train_p3_srr_h4_h8_fusion_wdice.ipynb](notebook/kaggle_train_p3_srr_h4_h8_fusion_wdice.ipynb).

Cả hai clone `https://github.com/FUAVX/RPD_Reproduce.git`, fetch nhánh
`codex/p3-srr`, rồi checkout detached tại `SOURCE_COMMIT` đầy đủ 40 ký tự.
Commit source: `9c9446c360ddaaff78db208c9cc3cb9b810a9b77`.
Commit source chứa model/config/tools được tạo trước commit chứa launcher;
notebook không cần chứa hash của chính commit của nó. Git sparse checkout bỏ
qua kết quả/checkpoint lịch sử. `p3_package_manifest.json` và source ZIP không
còn là input bắt buộc.

## Cách chạy Kaggle

1. Import mỗi notebook vào một session độc lập. Mỗi session mặc định cần
   **2 GPU** và bật Internet để cài dependencies/clone GitHub.
2. Attach PhenoBench. Root cần `train/images`, `train/semantics`, `val/images`,
   `val/semantics`, `test/images`. Loader gốc liệt kê cả test khi setup;
   WDice không yêu cầu plant/leaf instance masks. Nếu nhiều root phù hợp,
   điền `DATASET_DIR` rõ ràng.
3. Chạy cell dependencies; restart kernel nếu pip yêu cầu, rồi chạy lại
   các cell theo thứ tự. Giữ torch/torchvision/CUDA do Kaggle cấp.
4. Chạy cell clone/config. Giữ `SOURCE_COMMIT` đã khóa. Run luôn có token
   riêng trong `/kaggle/working/<SCENARIO>/<token>`.
5. Chạy preflight: normal import, factory, recipe, shape/input lẻ, backward,
   WDice và deploy diagnostic trên CPU; đọc mẫu train/val thật; GPU kiểm tra
   full batch train 768, backward + AdamW, validation theo shape thực tế.
   Xem peak memory trước khi chạy cell train. Lỗi preflight chặn train.
6. Chạy cell train sau khi preflight qua. Không chạy hai notebook trong cùng
   một session vì mặc định mỗi run dùng cả hai GPU.
7. Tải `<SCENARIO>_<token>_artifacts.zip` và lưu output Kaggle. Artifacts gồm
   manifest, active config, source diff/status, log, preflight và checkpoints.
   Cell cuối có thể đóng gói lại sau khi ngắt có kiểm soát. Runtime bị kill
   hoàn toàn có thể không chạy `finally`; giữ các checkpoint/output đã lưu.

Giữ recipe: 100 epochs; seed 1682409321; LR 2e-4; cosine; warmup 5;
weight decay 2e-4; FP32; batch 4/GPU; accumulate 1; WDice = weighted CE +
macro Soft Dice (lambda 1, epsilon 1e-6), weights [1.47, 5.06, 10.02].
YAML P3 chỉ khác P2 Ex7 ở `experiment`/`backbone`. Notebook ghi đè đường dẫn
dataset, workers và số devices để chạy trên Kaggle. Với hai GPU, effective
batch là 8. Nếu đổi GPU/batch/precision, dùng đối chứng B1 + WDice tương ứng
và lưu điều kiện thay đổi; không tự giảm batch hoặc đổi loss khi OOM.

Validation giữ native resolution của loader gốc, dù YAML khai báo
`center_crop: 1024`. Preflight dùng shape của mẫu val đầu tiên; chưa kiểm tra
tất cả ảnh hoặc bảo đảm VRAM cho toàn bộ dataset/DDP.

## Fresh và resume

Để trống `RESUME_CHECKPOINT`, `RESUME_CONFIG`, `RESUME_MANIFEST` để fresh.
Không tự tìm `last.ckpt` trong `/kaggle/input`.

Resume cần chỉ định đủ ba đường dẫn của cùng run. Notebook kiểm tra scenario,
backbone, recipe, hash config và source commit. Người chạy phải chọn đúng
checkpoint đi kèm manifest/config; không dùng checkpoint PA1 cho PA2.
Chỉ dùng checkpoint đáng tin cậy: Lightning 1.9 cần tải đầy đủ trạng thái
training, nên launcher bật `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` khi resume
trên PyTorch mới. Mỗi lần resume vẫn tạo output token riêng.

Mỗi manifest ghi repository, branch, commit thực tế, source diff/status,
active config + SHA256, GPU, effective batch và đường dẫn resume. Source phải
sạch ở commit đã khóa; đổi code cần review/commit và cập nhật pin. Sửa active
config sau setup buộc chuẩn bị run và preflight lại.

## Kiểm tra và giới hạn

Từ root repository, dùng môi trường có PyTorch, torchvision,
Lightning 1.9.5, torchmetrics 0.11.4, timm 0.9.16 và dependencies notebook:

```bash
python test_p3.py --output /tmp/p3_normal.json
python test_p3.py --core-only --output /tmp/p3_core.json
python test_wdice.py
python multi_metric_train.py --help
python tools/p3_notebook_checks.py
```

`--core-only` chỉ là kiểm tra cô lập khi thiếu training dependencies; không
thay thế normal import. Notebook luôn dùng normal import. CPU reports và
kiểm tra launcher được lưu trong `docs/p3/` sau khi chạy tại lần tích hợp này.

Số params dự kiến: B1/PA1 189827; PA2 333891; C0 292611. Deploy diagnostic
báo cả sai số tuyệt đối và sai số chuẩn hóa theo output random; chưa chứng
nhận absolute tolerance trên checkpoint đã train. Không có kết quả chất
lượng segmentation từ các tensor random này.

Chưa chạy GPU Kaggle, PhenoBench thật hoặc training dài trong lần tích hợp.
Weed IoU/mIoU đã thuộc training loop gốc; coverage theo vùng nhỏ và miss rate
cần đánh giá riêng sau train. Dùng B1 + cùng WDice làm đối chứng và cùng quy
tắc chọn checkpoint. Chưa có kết luận cải thiện, FLOPs hay latency GPU.
