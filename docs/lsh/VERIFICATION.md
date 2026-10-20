# Kiểm chứng triển khai 6-LSH

**Người thực hiện: PhuongTU**

## Repository và source

Remote đúng: `https://github.com/FUAVX/RPD_Reproduce.git`.
Trước khi sửa, checkout `/home/trandouyenphuong/RPD_Reproduce` ở `p2_ende`
`1f391562790f52aa7975904857e89b3da426efbb`; worktree SRR ở `f/srr`
`7b19c752e146559c7ae81fd3abce3fd92bdda040`. Cả hai không có local changes
tracked/untracked. Không có AGENTS.md áp dụng được tìm thấy ở project và các
ancestor đã kiểm tra. Đã fetch remote, kiểm tra branch `f/lsh` chưa tồn tại.

Worktree riêng: `/home/trandouyenphuong/RPD_LSH_PhuongTU`, branch `f/lsh`,
tạo từ `7b19c752e146559c7ae81fd3abce3fd92bdda040`. Chọn nền sau đối chiếu
source, không khẳng định f/srr là nhánh mới nhất của mọi hướng thử nghiệm.
B1/WDice/data/caller/P2 config ở nền này trùng byte với
`d77d518b8f650443216070fa76742db76e41b8ac`; các experiment khác không được
merge/cherry-pick để thay recipe. SRR được giữ làm source lịch sử nhưng LSH
chỉ kế thừa RepDWNet B1 và giữ down4 /16.

Source commit đã push trước launcher:
`e39b25018f27bf107a406d90270245ee80b0b233`.
Hai notebook khóa cùng full SHA này; commit chứa notebook có thể đọc bằng
`git log` trên f/lsh. Runtime notf hook kế thừa đúng nội dung artifact sửa
TensorBoard trong bàn giao, nay có source .py được quản lý bằng Git và được
copy ra ngoài checkout khi chạy. Không sửa các plan chuẩn.

## Những file được thêm/cập nhật

- Thêm `models/rpdnet/RepDWNetLSHL1.py`, `RepDWNetLSHL2.py`.
- Cập nhật có chọn lọc `models/__init__.py`: thêm import/constructor L1/L2.
- Thêm `configs/p3_scenarios/P3_LSH_L1_WDice.yaml`, `P3_LSH_L2_WDice.yaml`.
- Thêm `notebook/kaggle_train_p3_lsh_l1_wdice.ipynb`,
  `kaggle_train_p3_lsh_l2_wdice.ipynb`.
- Thêm `test_lsh.py`, `tools/lsh_launcher.py`, `tools/lsh_preflight.py`,
  `tools/lsh_ddp_smoke.py`, `tools/lsh_logger_smoke.py`,
  `tools/lsh_runtime/sitecustomize.py`, `tools/lsh_launcher_checks.py`,
  `tools/lsh_notebook_checks.py`.
- Thêm `README_LSH_PhuongTU.md`, tài liệu này và các JSON/environment.txt
  bằng chứng trong `docs/lsh/`.

Không sửa baseline model/blocks/loss/loader/trainer, YAML P2 hoặc source SRR.
Danh sách byte-identical/hash được lưu trong `source_integrity.json`.
Hai YAML mới chỉ khác experiment/backbone so với P2 Ex7. Optimizer thực
trong training caller là Adam; preflight dùng cùng class/options.

## Đã chạy

Môi trường cô lập `/home/trandouyenphuong/.venvs/rpd-p3`, Python 3.11.16,
PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, Lightning 1.9.5,
torchmetrics 0.11.4, timm 0.9.16, TensorBoard 2.21.0, NumPy 1.26.4.
`environment.txt` ghi full package versions; `pip check` qua.

1. `python test_lsh.py --output /tmp/lsh-checks/core.json` — qua normal import,
   factory, recipe equality; Conv1×1/bias/channels và không bỏ down4;
   output `[2,3,64,64]` / `[2,3,65,79]`; so công thức mean logits chính xác;
   shared initial weights và /1 feature readout khớp L0 cùng weights.
   Một WDice có mask ignore255, loss và mọi gradient finite. Bốn test WDice
   gốc chạy lại trên implementation không đổi. AST kiểm tra callback
   `val_weed_iou/max` và optimizer Adam thực trong caller.
2. Cùng test dùng PNG fixture qua PDC gốc: train crop768, val native65×79,
   raw3→1/4→2. Fixture không phải dataset PhenoBench. Shape/stats lưu trong
   `core_checks.json`.
3. DDP riêng từng model: `torch.distributed.run --standalone --nnodes=1
   --nproc-per-node=2 tools/lsh_ddp_smoke.py --device cpu ...` — hai process,
   mỗi process hai Adam steps, backend Gloo, `find_unused_parameters=False`.
   Dữ liệu khác từng rank; mọi gradient finite và hash tham số đồng bộ sau
   bước optimizer. Cả hai rank kế thừa startup hook/notf. Bằng chứng trong
   `ddp_cpu_l1.json` và `ddp_cpu_l2.json`.
4. `tools/lsh_logger_smoke.py` dưới PYTHONPATH chứa startup hook và
   `P3_TENSORBOARD_NO_TF=1` — SummaryWriter ghi/đọc scalar và grouped scalar,
   Lightning TensorBoardLogger log_hyperparams/log_metrics/save/finalize,
   descendant Python giữ flag/hook và backend tensorflow_stub. Không import
   TensorFlow/JAX thật. Backend lazy được resolve trước khi kiểm tra tên.
   Kết quả trong `logger_checks.json`.
5. `tools/lsh_launcher_checks.py` — Git fixture, active config/effective batch,
   source/config/hook guards, logger subprocess thật, chặn GPU preflight trên
   CPU và chặn train, kiểm tra resume source/scenario/config/checkpoint hash,
   CLI fresh/resume với process train mocked, artifact loại source.
   `launcher_checks.json` không được diễn giải thành train đã chạy.
6. `tools/lsh_notebook_checks.py --remote ...` — thực thi cell clone/config
   của cả hai notebook từ GitHub ở source SHA đã push; logger cell thật;
   CPU bị chặn ở GPU preflight như mong đợi; train gate từ chối; package
   artifacts. Chỉ attach cấu trúc thư mục fixture rỗng, không dataset thật.
   Kết quả trong `notebook_checks.json`.
7. Cả hai notebook được nbformat validate; compile mọi code cell; outputs và
   execution_count được làm sạch. Model nằm trong source .py, không nhúng
   class model vào notebook. CLI/import `multi_metric_train.py --help` và
   `tools/lsh_preflight.py --help` qua; compileall file mới qua.
8. Git whitespace check qua với cr-at-eol cho factory CRLF sẵn có.

| Model | Params | Head params |
|---|---:|---:|
| L0 | 189827 | 51 |
| L1 | 189878 | 102 |
| L2 | 189977 | 201 |

Deploy probe random cho hai model có normalized max error <1e-5; báo cả
absolute error trong JSON. Logits random chưa hiệu chỉnh có biên độ lớn;
không xác nhận absolute tolerance trên checkpoint đã train và không coi
loss trên tensor random là kết quả chất lượng.

## Chưa xác minh

- Không có CUDA/GPU hoặc PhenoBench thật ở local. Chưa chạy GPU batch4 crop768,
  native validation memory, GPU/NCCL DDP, hoặc toàn dataset.
- Notebook có per-model subprocess cho các kiểm tra trên trước train. GPU
  preflight lặp sample train/val đầu tiên tới batch cấu hình; không chứng
  nhận toàn dataset hay full-batch DDP memory. DDP probe dùng synthetic64.
- Chưa chạy checkpoint resume thật với Lightning; CLI/provenance guards đã
  kiểm tra bằng fixture/mocked dispatch. Khi hard-kill chưa có digest đóng
  gói, người chạy phải giữ đúng checkpoint trong bundle run gốc.
- Chưa chứng minh bản sửa TensorBoard chạy training dài trên Kaggle. Local
  logger/DDP-startup checks qua là bằng chứng hẹp hơn.
- Chưa khởi chạy train dài L0/L1/L2; chưa có Weed IoU/mIoU so sánh, coverage
  vùng nhỏ/miss rate, latency/FLOPs/GPU memory hoặc cải thiện được chứng minh.
- Ngưỡng vùng nhỏ, danh sách seed bổ sung, ngân sách latency/memory còn mở
  theo plan. Quy tắc checkpoint cho so sánh này đã chọn trước train là
  best val_weed_iou/max cho cả ba run; không thay callback baseline.
