# Preprocessing Pipeline
### Skin Disease Classification · Data Preprocessing Guide

---

## Mục lục

- [Tổng quan](#-tổng-quan)
- [Cấu trúc notebook](#-cấu-trúc-notebook)
- [Luồng dữ liệu](#-luồng-dữ-liệu)
- [Chi tiết từng bước](#-chi-tiết-từng-bước)
- [Các quyết định thiết kế quan trọng](#-các-quyết-định-thiết-kế-quan-trọng)
- [Output của pipeline](#-output-của-pipeline)
- [Lỗi thường gặp](#-lỗi-thường-gặp)

---

## Tổng quan

Notebook `01_preprocessing.ipynb` thực hiện **toàn bộ quá trình chuẩn bị dữ liệu** trước khi đưa vào huấn luyện mô hình CNN. Đây là bước quan trọng nhất trong pipeline — một preprocessing sai có thể làm hỏng toàn bộ quá trình training dù model tốt đến đâu.

**Mục tiêu:**
- Phân tích và hiểu rõ đặc điểm dataset
- Tách train/val không gây data leakage
- Chuẩn hóa ảnh đúng cách cho pretrained models
- Xử lý class imbalance ở tầng data
- Xuất ra DataLoaders sẵn sàng cho training

**Dataset:** Skin Disease — 22 classes, 13,898 ảnh train, 1,546 ảnh test  
**Source:** [KaggleHub — pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

---

## 📂 Cấu trúc notebook

```
01_preprocessing.ipynb
│
├── Cell 1  — Import thư viện
├── Cell 2  — Config & hằng số
├── Cell 3  — Seed toàn cục (Reproducibility)
├── Cell 4  — Load raw dataset & EDA cơ bản
├── Cell 5  — Phân tích phân phối class (Label Distribution)
├── Cell 6  — Phân tích kích thước ảnh
├── Cell 7  — Train/Val Split (không data leakage)
├── Cell 8  — Định nghĩa Transforms
├── Cell 9  — Tạo Dataset với transform
├── Cell 10 — Class Imbalance: Class Weights
├── Cell 11 — Class Imbalance: WeightedRandomSampler
├── Cell 12 — Tạo DataLoaders
├── Cell 13 — Sanity Checks
└── Cell 14 — Tổng kết & Summary
```

---

## Luồng dữ liệu

```
Ảnh trên disk (PNG/JPEG, nhiều kích thước)
          │
          ▼
  ImageFolder (raw, không transform)
  → Đọc metadata: labels, class_names
  → Phân tích phân phối, kích thước
          │
          ▼
  Index Split (90/10, seed=42)
  → train_idx : 12,508 indices
  → val_idx   :  1,390 indices
  → Đảm bảo KHÔNG overlap
          │
          ├─────────────────────┐
          ▼                     ▼
  ImageFolder                ImageFolder
  + train_transform          + val_transform
  → Subset(train_idx)        → Subset(val_idx)
  = train_dataset            = val_dataset
          │
          ▼
  WeightedRandomSampler
  → Oversample minority classes
  → replacement=True
          │
          ▼
  DataLoader
  → Batch size = 32
  → pin_memory = True (GPU)
  → drop_last  = True (train)
          │
          ▼
  Tensor (32, 3, 224, 224)
  dtype  : float32
  range  : ≈ [-2.5, 2.5]
  device : CUDA / CPU
          │
          ▼
        MODEL
```

---

## Chi tiết từng bước

### Bước 1 — Import thư viện

Chỉ import những gì preprocessing cần — tách biệt hoàn toàn với thư viện training:

```python
# Thư viện chuẩn
import os, random
from collections import Counter

# Tính toán
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch — chỉ phần data
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# Sklearn — tính class weights
from sklearn.utils.class_weight import compute_class_weight
```

---

### Bước 2 — Config

Tất cả tham số tập trung một chỗ theo nguyên tắc **Single Source of Truth**:

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `IMAGE_SIZE` | 224 | Chuẩn ImageNet cho ResNet/DenseNet/EfficientNet |
| `BATCH_SIZE` | 32 | Phù hợp với RTX 3050 6GB VRAM |
| `VAL_RATIO` | 0.1 | 10% train làm validation (~1,390 ảnh) |
| `NUM_WORKERS` | 4 | Song song hóa đọc ảnh (Windows: đặt 0 nếu lỗi) |
| `SEED` | 42 | Cố định cho reproducibility |

---

### Bước 3 — Seed toàn cục

Cố định **4 nguồn ngẫu nhiên** để kết quả lặp lại được:

```python
random.seed(seed)          # Python built-in
np.random.seed(seed)       # NumPy
torch.manual_seed(seed)    # PyTorch CPU
torch.cuda.manual_seed_all(seed)  # PyTorch GPU

torch.backends.cudnn.deterministic = True  # Chậm hơn ~5% nhưng deterministic
torch.backends.cudnn.benchmark     = False # Tắt auto-tuner
```

---

### Bước 4 — Load raw & EDA

Load `ImageFolder` **không transform** để đọc metadata nhanh:

```python
full_raw = datasets.ImageFolder(TRAIN_DIR)  # không transform
```

Phân tích imbalance:
- **Max/Min ratio: 6.6x** (Unknown_Normal: 1,651 vs Candidiasis: 248)
- 9/22 class nằm dưới mức trung bình (632 ảnh)
- → Cần: WeightedRandomSampler + FocalLoss

---

### Bước 5 — Phân tích kích thước ảnh

Lấy mẫu 20 ảnh/class để phân tích:
- Ảnh có **nhiều kích thước khác nhau** → bắt buộc resize
- Nhiều ảnh không phải hình vuông → resize thẳng gây méo
- → Giải pháp: `Resize(256)` → `CenterCrop(224)`

---

### Bước 6 — Train/Val Split (quan trọng nhất)

**Vấn đề data leakage phổ biến:**
```python
# SAI — val set bị augment
dataset = ImageFolder(root, transform=train_transform)
train, val = random_split(dataset, [0.9, 0.1])
```

**Cách đúng trong pipeline này:**
```python
# ĐÚNG — split index trước, gán transform sau
indices   = list(range(n))
random.shuffle(indices)
train_idx = indices[:split]   # 90%
val_idx   = indices[split:]   # 10%

# Hai ImageFolder riêng biệt với transform khác nhau
train_ds = Subset(ImageFolder(TRAIN_DIR, transform=train_transform), train_idx)
val_ds   = Subset(ImageFolder(TRAIN_DIR, transform=val_transform),   val_idx)

# Kiểm tra không overlap
assert len(set(train_idx) & set(val_idx)) == 0
```

---

### Bước 7 — Transforms

**Train transform** (có augmentation):

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),    # tổn thương trái/phải tương đương
    transforms.RandomRotation(degrees=10),      # tay nghiêng nhẹ khi chụp
    transforms.ColorJitter(                     # ánh sáng, thiết bị khác nhau
        brightness=0.2, contrast=0.2,
        saturation=0.1, hue=0.03               # hue nhỏ — màu tổn thương quan trọng
    ),
    transforms.GaussianBlur(kernel_size=3),    # camera điện thoại không focus tốt
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # BẮT BUỘC với pretrained
    transforms.RandomErasing(p=0.1),           # tóc/quần áo che khuất
])
```

**Val/Test transform** (deterministic, không augment):
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),                # không ngẫu nhiên
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**Tại sao augmentation nhẹ?**

Dataset được chụp trong điều kiện lý tưởng (góc chuẩn, ánh sáng tốt). Augmentation nhằm mô phỏng **gap giữa dataset và thực tế deployment** — không phải để làm khó model. Các transform bị loại bỏ:

| Transform loại bỏ | Lý do |
|-------------------|-------|
| `RandomVerticalFlip` | Không ai chụp ảnh bệnh lộn ngược |
| `RandomRotation(30°)` | Quá xa thực tế với ảnh đã chuẩn |
| `RandomAffine` mạnh | Làm méo tổn thương |
| `ColorJitter` mạnh | Màu tổn thương là thông tin chẩn đoán |

**ImageNet Normalization:**
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```
Bắt buộc với pretrained models — dùng sai stats làm giảm ~10% accuracy do lệch distribution mà weights đã học.

---

### Bước 8 — Class Imbalance

**Hai tầng xử lý:**

**Tầng 1 — WeightedRandomSampler (tầng data):**
```python
# Công thức: weight[c] = total / (n_classes × count[c])
class_weights  = compute_class_weight('balanced', ...)
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True   # ảnh minority được chọn lại nhiều lần
)
```

Lưu ý quan trọng: **Sampler chạy trước, augmentation chạy sau**.
- Sampler chọn index → DataLoader đọc ảnh → train_transform apply
- Mỗi lần ảnh được chọn lại, augmentation tạo ra version khác nhau
- → Oversampling + Augmentation bổ sung cho nhau

**Tầng 2 — FocalLoss (tầng loss, dùng khi training):**
```python
FL = -α_t × (1 - p_t)^γ × log(p_t)
# γ=2: focus vào hard examples
# α_t: class weights từ tầng 1
```

---

### Bước 9 — DataLoaders

| Tham số | Train | Val/Test | Lý do |
|---------|-------|----------|-------|
| `sampler` | WeightedRandomSampler | None | Chỉ train cần oversample |
| `shuffle` | False (sampler tự shuffle) | False | Deterministic evaluation |
| `pin_memory` | True | True | Transfer RAM→GPU nhanh hơn 2× |
| `drop_last` | True | False | Tránh batch cuối không đủ size |

---

### Bước 10 — Sanity Checks

7 assertions bắt buộc trước khi train:

```python
assert tuple(batch.shape) == (BATCH_SIZE, 3, 224, 224)  # shape đúng
assert batch.dtype == torch.float32                       # dtype đúng
assert not torch.isnan(batch).any()                       # không NaN
assert not torch.isinf(batch).any()                       # không Inf
assert batch.min() >= -3.5 and batch.max() <= 3.5         # range hợp lý
assert len(set(train_idx) & set(val_idx)) == 0            # không overlap
assert torch.equal(v_batch1, v_batch2)                    # val deterministic
```

---

## Các quyết định thiết kế quan trọng

### 1. Tại sao không dùng `random_split`?

`random_split` gán transform cho toàn bộ dataset trước khi split → val set bị augment → metrics không ổn định. Pipeline này tạo hai `ImageFolder` riêng biệt với transform khác nhau, dùng chung index.

### 2. Tại sao Macro F1 thay vì Accuracy?

Với imbalance 6.6x, model predict `Unknown_Normal` cho tất cả đạt accuracy 30% nhưng recall minority = 0%. Macro F1 tính đều cho tất cả 22 class, phản ánh đúng chất lượng thực.

### 3. Tại sao giữ `Unknown_Normal` có noise?

Phản ánh thực tế deployment — không phải ảnh nào người dùng gửi lên cũng là ảnh bệnh da. Class này đóng vai trò safety valve. Noise được giảm thiểu bởi WeightedSampler (weight = 0.38).

### 4. Tại sao watermark không cần xử lý?

Watermark xuất hiện đồng đều trên tất cả class → không tạo bias giữa các class. `RandomErasing` giúp model learn to ignore vùng cố định. Xác nhận qua Grad-CAM: heatmap xanh tại vị trí watermark → model đang ignore đúng cách.

---

## Output của pipeline

Sau khi chạy xong, các objects sau sẵn sàng cho training:

```python
train_loader  # DataLoader — WeightedSampler, BATCH_SIZE=32, drop_last=True
val_loader    # DataLoader — shuffle=False, deterministic
test_loader   # DataLoader — shuffle=False, deterministic
class_names   # List[str]  — tên 22 class theo alphabet
NUM_CLASSES   # int        — 22
cw_tensor     # Tensor     — class weights để khởi tạo FocalLoss
IMAGENET_MEAN # List[float]— để unnormalize khi visualize
IMAGENET_STD  # List[float]— để unnormalize khi visualize
```

**Spec của tensor đầu ra:**
```
Shape  : (32, 3, 224, 224)
Dtype  : torch.float32
Range  : ≈ [-2.5, 2.5]
Layout : (Batch, Channel, Height, Width)
```

---

## ❗ Lỗi thường gặp

### `NameError: name 'gridspec' is not defined`
Import bị đặt sai thứ tự. Đảm bảo `from matplotlib import gridspec` nằm **trước** dòng dùng nó.

### `RuntimeError: DataLoader worker exited unexpectedly`
Windows multiprocessing issue. Đặt `NUM_WORKERS = 0`.

### `AssertionError` trong Sanity Checks
Chạy lại toàn bộ notebook từ đầu (Kernel → Restart & Run All) để đảm bảo các biến được khởi tạo đúng thứ tự.

### Val deterministic check thất bại
`val_transform` đang có augmentation ngẫu nhiên. Kiểm tra lại — val transform chỉ được có `Resize`, `CenterCrop`, `ToTensor`, `Normalize`.

---

## Thống kê dataset

| Class | Train | Class | Train |
|-------|-------|-------|-------|
| Unknown_Normal | 1,651 | Moles | 361 |
| Benign_tumors | 1,093 | Seborrh_Keratoses | 455 |
| Eczema | 1,010 | Vasculitis | 461 |
| Tinea | 923 | Vascular_Tumors | 543 |
| Psoriasis | 820 | DrugEruption | 547 |
| Actinic_Keratosis | 748 | Lichen | 553 |
| Vitiligo | 714 | Warts | 580 |
| SkinCancer | 693 | Acne | 593 |
| Bullous | 504 | Infestations_Bites | 524 |
| **Candidiasis** ⚠️ | **248** | **Rosacea** ⚠️ | **254** |
| **Lupus** ⚠️ | **311** | **Sun_Sunlight_Damage** ⚠️ | **312** |

⚠️ = Minority class (< 400 ảnh) — cần WeightedSampler + FocalLoss

**Imbalance ratio: 6.6x** (Unknown_Normal / Candidiasis)
