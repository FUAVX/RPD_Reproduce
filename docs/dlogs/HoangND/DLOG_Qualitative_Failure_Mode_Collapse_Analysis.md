# Development Log: Case Study — Hiện Tượng Sụp Đổ Dự Đoán (Prediction Collapse) Trên Nền Đất Trống Nắng Gắt & Nghịch Lý Thuật Toán Xếp Hạng Định Tính

| Metadata | Chi tiết |
|---|---|
| **Mã tài liệu** | `DLOG-2026-10-11-HOANGND-01` |
| **Tác giả** | HoangND — Lead Learning Method / Baseline |
| **Dự án** | WeedSegment_DPL302m-FALL26 / RPD_Reproduce |
| **Ngày ghi nhận** | 2026-10-11 |
| **Chủ đề** | Qualitative Visualizer Analysis — Single-Image Normalization Collapse & Ranking Selection Bias |
| **Trạng thái** | Đã cô lập nguyên nhân khoa học, sửa mã nguồn và bổ sung chốt chặn tự động (`is_collapsed` guard) |

---

## 1. Bối cảnh & Hiện tượng Quan sát được

Khi thực thi script trực quan hóa so sánh đa mô hình (`scripts/visualize_multi_model_comparison.py`) trên tập Validation PhenoBench (769 ảnh) nhằm xuất dải ảnh đối soát 7 khung hình (7-Panel Strips: RGB, Ground Truth, B0, B1, P2_ex1, P2_ex2, Discrepancy Map), chúng tôi quan sát thấy một sự phân cực kỳ lạ giữa các hàng ảnh:

```
[Hiển thị]: rank01_05-26_00031_P0034282_7panel.png  ---> [HOÀN HẢO] Đất, cây, cỏ sắc nét
[Hiển thị]: rank01_05-26_00220_P0034121_7panel.png  ---> [SỤP ĐỔ] Toàn xanh, toàn đỏ, vân sọc
[Hiển thị]: rank02_05-26_00199_P0034121_7panel.png  ---> [SỤP ĐỔ] Toàn xanh, toàn đỏ, vân sọc
[Hiển thị]: rank02_05-26_00205_P0034016_7panel.png  ---> [HOÀN HẢO] Đất, cây, cỏ sắc nét
```

### Bảng đối chiếu số liệu pixel đo đạc trực tiếp trên tensor:

| Khung hình (Panel) | Hàng 1 (`05-26_00031_P0034282`) | Hàng 2 (`05-26_00220_P0034121`) | Bản chất hiện tượng Hàng 2 |
|---|---|---|---|
| **Ground Truth (GT)** | Đất 80.5%, Cây 18.2%, Cỏ 1.3% | **Đất 98.19%**, Cây 1.53%, Cỏ 0.27% | Đất trống chiếm gần như tuyệt đối |
| **B0 (RPDNet PDC)** | Đất 89.9%, Cây 9.7%, Cỏ 0.4% | **Đất 0.00%**, Cây 60.14%, Cỏ 39.86% | **Mất đất 100%**, vân sọc đỏ-xanh phủ kín |
| **B1 (RepDWNet Base)**| Đất 93.5%, Cây 5.8%, Cỏ 0.7% | **Đất 0.00%**, Cây 100.00%, Cỏ 0.00% | **Mất đất 100%**, bão hòa 100% Cây (xanh lá) |
| **P2_ex1 (2DW+4PW)** | Đất 92.6%, Cây 6.8%, Cỏ 0.6% | **Đất 0.00%**, Cây 0.00%, Cỏ 100.00% | **Mất đất 100%**, bão hòa 100% Cỏ (đỏ tươi) |
| **P2_ex2 (4DW+2PW)** | Đất 93.8%, Cây 5.2%, Cỏ 1.0% | **Đất 0.00%**, Cây 0.19%, Cỏ 99.81% | **Mất đất 100%**, bão hòa 99.8% Cỏ (đỏ tươi) |

### Câu hỏi cốt lõi:
1. **Trọng số có bị nạp sai không?** $\rightarrow$ **Không!** Nếu nạp sai trọng số (random weights), Hàng 1 và Hàng 4 cũng phải bị hỏng. Nhưng Hàng 1 và Hàng 4 phân đoạn viền lá cây, nhận diện nền đất và khóm cỏ cực kỳ chính xác.
2. **Vậy tại sao ở Hàng 2 và Hàng 3, cả 4 mô hình độc lập đều không dự đoán được dù chỉ 1 pixel Đất (Soil = 0%), mà lại sụp đổ thành những mảng màu đặc xanh/đỏ?**

---

## 2. Phân Tích Bản Chất Kỹ Thuật & Toán Học (Root Cause Analysis)

Hiện tượng này là kết quả cộng hưởng của **2 cơ chế toán học**:

```
[Ảnh Đất Trống Đồng Nhất 98.2%] 
       │
       ▼
[SingleImageNormalizer: Trừ Mean, Chia Std] 
       │
       ▼ (Toàn bộ nền đất bị đưa về quanh mức 0)
[Mất Tín Hiệu Kết Cấu Đất (Zero-Centered Loss of Texture)]
       │
       ▼
[Logits Bị Thả Trôi Theo Bias Cuối Của Classification Head]
       │
       ├─► B1: Bias Kênh Cây Lớn Hơn ────────► Bão hòa 100% Xanh (Crop)
       ├─► P2: Bias Kênh Cỏ Lớn Hơn ────────► Bão hòa 100% Đỏ (Weed)
       └─► B0: Vi Sai PDC Khuếch Đại Nhiễu ──► Vân Sọc Lưỡng Cực Đỏ - Xanh
```

### 2.1 Cơ chế chuẩn hóa từng ảnh (`SingleImageNormalizer`)
Công thức tiền xử lý được cấu hình trong `configs/*_scenarios/*.yaml`:
$$t_{\text{norm}}(c, y, x) = \frac{t(c, y, x) - \mu_{\text{img}}(c)}{\sigma_{\text{img}}(c) + \epsilon}$$

* **Khi ảnh có cây củ cải lớn (Hàng 1, 4):**
  Lá cây (hấp thụ ánh sáng Đỏ/Xanh dương, phản xạ Xanh lá mạnh) và Đất cát (phản xạ Đỏ/Cam cao) tạo ra phân phối màu lưỡng đỉnh (bimodal distribution). Độ lệch chuẩn $\sigma$ lớn ($\sigma_R \approx 50.1, \sigma_G \approx 42.9, \sigma_B \approx 35.0$). Quá trình chuẩn hóa giữ nguyên được biên độ tương phản giữa cây và đất.
* **Khi ảnh gần như toàn đất trống khô cằn nắng gắt (Hàng 2, 3 - thửa ruộng `P0034121`):**
  Đất chiếm tới $98.19\%$ diện tích toàn ảnh.
  $$\mu_{\text{img}}(c) \approx \text{Màu sắc trung bình của nền đất}$$
  Phép trừ $(t - \mu)$ **triệt tiêu toàn bộ màu đất về mức giá trị xấp xỉ 0**:
  $$\text{Feature}_{\text{soil}}(c, y, x) \approx 0$$
  Độ biến thiên $\sigma$ của ảnh lúc này chỉ là độ gồ ghề của sỏi đá li ti và nhiễu sensor.

### 2.2 Hiện tượng sụp đổ kích hoạt mạng nơ-ron (Prediction Collapse)
Khi một tensor kích thước $1024 \times 1024$ có giá trị xấp xỉ $0$ trên toàn không gian đi qua mạng tích chập sâu với các hàm kích hoạt ReLU:
$$\text{ReLU}(W \cdot 0 + b) = \max(0, b)$$

1. **Mạng mất hoàn toàn tín hiệu đặc trưng kết cấu không gian của đất.** Không có gradient màu sắc để kích hoạt các bộ lọc biên dạng.
2. **Logits đầu ra tại tầng phân loại cuối ($1 \times 1$ Conv) bị chi phối hoàn toàn bởi Bias:**
   $$\text{Logits}(c, y, x) \approx \text{Bias}_c + \text{Noise}$$
   * **Mô hình B1 (RepDWNet 4DW+4PW):** Trong quá trình huấn luyện Run 5, bias của lớp Cây trồng hơi nhỉnh hơn so với Đất và Cỏ $\rightarrow \text{argmax}$ gán toàn bộ $1,048,561$ pixel thành **Class 1 (Cây trồng - Xanh lá 100%)**.
   * **Mô hình P2_ex1 & P2_ex2:** Quá trình huấn luyện tinh chỉnh với số nhánh giảm làm bias của lớp Cỏ dại hơi nhỉnh hơn $\rightarrow \text{argmax}$ gán toàn bộ $1,048,576$ pixel thành **Class 2 (Cỏ dại - Đỏ tươi 100%)**.
   * **Mô hình B0 (RPDNet PDC Baseline):** Do B0 chứa các toán tử vi sai hướng tâm PDC ($f(x) - f(x_0)$), khi gặp nhiễu vi mô của hạt cát trên nền zero-mean, toán tử vi sai khuếch đại nhiễu tần số cao, tạo ra mạng lưới vân sọc lưỡng cực đỏ - xanh liên tục đảo chiều trên khắp mặt đất.

---

## 3. Nghịch Lý Thuật Toán Xếp Hạng (Ranking Selection Bias Paradox)

Một câu hỏi quan trọng: **Nếu trong 769 ảnh của tập Val, chỉ có 1-2 ảnh bị hiện tượng này, tại sao nó lại được chọn lên Rank 01 và Rank 02 của Báo cáo?**

Nguyên nhân nằm ở công thức tự động lọc ảnh của Category 1:
```python
# Tiêu chí chọn ảnh khử bóng râm nắng gắt:
"1_sunny_shadow_rejection": {
    "filter": lambda r: r["lighting"] == "Sunny II",
    "sort_key": lambda r: r["fp_b0_suppressed"],
}
```

Trong đó, `fp_b0_suppressed` đếm số pixel mà B0 báo động giả cỏ dại trên đất nhưng B1 không báo động:
$$\text{fp\_b0\_suppressed} = \sum_{y,x} \Big( \text{GT} \neq \text{Weed} \Big) \land \Big( \text{Pred}_{B0} == \text{Weed} \Big) \land \Big( \text{Pred}_{B1} \neq \text{Weed} \Big)$$

* **B0 bị lỗi vân sọc**: Đoán bừa **417,985 pixel đất là Cỏ dại**.
* **B1 bị lỗi bão hòa Cây**: Đoán toàn bộ ảnh là Cây trồng $\rightarrow$ **0 pixel Cỏ dại**.
* **Thuật toán chấm điểm:**
  $$\text{Score} = 417,985 - 0 = 417,985 \text{ điểm!}$$
* **Hậu quả:** Trên một bức ảnh bình thường, B1 chỉ khử được từ $500$ đến $2,500$ pixel cỏ giả của B0. Nhưng bức ảnh bị sụp đổ này đạt điểm số **417,985 điểm (kỷ lục tuyệt đối)**.
* Thuật toán máy móc đã đánh giá đây là "bức ảnh thể hiện sự vượt trội vĩ đại nhất của B1 trước B0", và lập tức đưa nó lên **Rank 01** và **Rank 02** để hiển thị ngay đầu notebook!

---

## 4. Giải Pháp Kỹ Thuật Đã Triển Khai

Để xử lý triệt để cả 2 mặt của vấn đề (chuẩn hóa chính xác và chống thiên lệch xếp hạng):

### 4.1 Đồng bộ chuẩn hóa dữ liệu với Codebase gốc
Thay vì tự viết công thức chuẩn hóa bằng tay trong script visualizer, chuyển sang gọi trực tiếp class `SingleImageNormalizer` được định nghĩa trong `RPD/datasets/augmentations_normalize.py`:
```python
from datasets.augmentations_normalize import SingleImageNormalizer
from torchvision import transforms

img_normalizer = SingleImageNormalizer()
to_tensor = transforms.ToTensor()

img_tensor = to_tensor(rgb_pil)
t = img_normalizer.normalize(img_tensor).unsqueeze(0).to(device)
```
Điều này đảm bảo việc tính toán các thống kê kênh và hằng số $\epsilon = 10^{-17}$ hoàn toàn đồng nhất với pha huấn luyện.

### 4.2 Bổ sung bộ lọc phát hiện sụp đổ mô hình (`is_collapsed` Guard)
Trong hàm `run_multi_model_pipeline`, bổ sung bộ kiểm tra tính toàn vẹn của không gian phân đoạn:
```python
total_pixels = gt_np.size
is_collapsed = False
for mname in models.keys():
    p_counts = np.bincount(preds[mname].ravel(), minlength=3)
    # 1. Bão hòa >85% diện tích thành 1 lớp trong khi GT không phải vậy
    if (p_counts[1] > 0.85 * total_pixels and crop_pixels < 0.50 * total_pixels) or \
       (p_counts[2] > 0.85 * total_pixels and weed_pixels < 0.50 * total_pixels):
        is_collapsed = True
        break
    # 2. Lớp Đất (Soil = 0) bị xóa sổ hoàn toàn (<1%) trong khi GT có đất >50%
    if p_counts[0] < 0.01 * total_pixels and (gt_np == 0).sum() > 0.50 * total_pixels:
        is_collapsed = True
        break
```

### 4.3 Cập nhật điều kiện lọc cho các danh mục so sánh
Tất cả các bộ chọn ảnh tiêu biểu bắt buộc phải thỏa mãn:
1. `not r["is_collapsed"]`: Mô hình không bị sụp đổ bão hòa.
2. `r["weed_pixels"] >= 200`: Ảnh phải có thực thể cỏ dại thực tế trên thực địa.
3. `r["stats"]["B1"]["iou"] > 10.0%`: Mô hình phân đoạn hợp lệ và có nghĩa nông nghiệp.

---

## 5. Kết Luận & Ý Nghĩa Nghiên Cứu

1. **Về mặt kỹ thuật mô hình:**
   * Không có lỗi hỏng trọng số (weights loading). Các mô hình đều hoạt động rất tốt trên các điều kiện thực địa nông nghiệp tiêu chuẩn (có cây, có cỏ, có đất).
   * Thửa ruộng `P0034121` là một **Edge Case / Failure Mode** kinh điển của phương pháp chuẩn hóa `SingleImageNormalizer` khi áp dụng trên các ảnh vi mô thiếu hoàn toàn vật thể tiền cảnh (Foreground-deprived homogeneous background).
2. **Về mặt phương pháp luận đánh giá:**
   * Không bao giờ nên dùng các hàm mục tiêu chênh lệch ($\Delta_{\text{FP}}$) một cách máy móc mà không có **Sanity Check** về tính toàn vẹn của phân phối nhãn.
   * Việc bổ sung chốt chặn `is_collapsed` giúp báo cáo định tính (Qualitative Gallery) phản ánh đúng 100% năng lực nhận diện cạnh lá, kháng bóng râm và an toàn phun thuốc thực tế của B1, P2_ex1 và P2_ex2.
