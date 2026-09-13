# PLAN — Đánh giá độ bền của static Android malware detection dưới obfuscation

Dataset: CICMalDroid 2020 (17.341 APK, 5 lớp: Benign, Adware, Banking, SMS, Riskware)
Môi trường: Google Colab, điều khiển từ VSCode qua remote kernel
Mục tiêu: không phải đẩy accuracy lên cao hơn, mà **đo xem accuracy hiện tại sụp đổ ở đâu và vì sao**

---

## 0. Ràng buộc phải tôn trọng

| Ràng buộc | Hệ quả lên thiết kế |
|---|---|
| Dataset chỉ trải 12 tháng (12/2017–12/2018), không có timestamp mẫu | **Bỏ temporal split ở dự án này.** Trục drift để sang AndroZoo/LAMDA ở giai đoạn sau |
| CSV/JSON của CIC chỉ tồn tại cho 17.341 APK gốc | **APK đã obfuscate không có dòng CSV tương ứng, và CopperDroid không deploy lại được.** Phải tự trích feature tĩnh bằng Androguard — đây là điều kiện cần để thí nghiệm tồn tại, không phải bước tối ưu. CSV gốc vẫn giữ làm baseline đối chiếu với literature |
| Benign (~4.039) ít hơn malware (~12.623), khác nguồn | Báo cáo theo macro-F1 + per-class, không dùng accuracy tổng |
| Colab timeout ~12h, disk reset mỗi session | Mọi bước dài phải checkpoint xuống Drive theo lô |
| Obfuscapk rất chậm (~30–90s/APK/kỹ thuật) | **Chỉ obfuscate tập test đã subsample**, không bao giờ chạy trên 17k APK |

**Tính toán chi phí obfuscation** (để không ai thử chạy full):
- 17.341 APK × 6 nhóm kỹ thuật × 45s ≈ 1.300 giờ → không khả thi
- 500 APK test × 6 nhóm × 45s ÷ 4 worker ≈ **9–10 giờ** → chia 2 session Colab, khả thi

---

## 1. Cấu trúc repo

```
apk-obfuscation-robustness/
├── notebooks/
│   └── colab_driver.ipynb        # cell tải dataset + mount Drive
├── src/
│   ├── download.py               # tải + giải nén CICMalDroid
│   ├── features/
│   │   ├── extract.py            # Androguard → feature dict per APK
│   │   ├── groups.py             # định nghĩa 5 nhóm feature G1–G5
│   │   └── vectorize.py          # dict → sparse matrix, fit trên train
│   ├── obfuscate.py              # wrapper Obfuscapk, chạy song song
│   ├── train.py                  # baseline ML
│   ├── evaluate.py               # eval sạch + eval obfuscated
│   └── matrix.py                 # sinh ma trận kỹ thuật × nhóm feature
├── data/                         # gitignore, symlink sang /content/drive
├── results/
└── PLAN.md
```

---

## 2. Phase 0 — Setup Colab (30 phút)

Cell trong `colab_driver.ipynb`:

```python
# Mount Drive để checkpoint
from google.colab import drive
drive.mount('/content/drive')
WORK = '/content/drive/MyDrive/apk-robustness'

# Java + Android build-tools cho Obfuscapk
!apt-get -qq install -y openjdk-17-jdk-headless zipalign apksigner
!pip -q install androguard==4.1.2 obfuscapk scikit-learn pandas pyarrow tqdm
```

**Kiểm tra ngay sau khi cài:** chạy Obfuscapk trên 1 APK mẫu với kỹ thuật `Rebuild`, xác nhận file output tồn tại và `apksigner verify` pass. Nếu bước này hỏng thì mọi thứ sau đều vô nghĩa — đừng đi tiếp.

---

## 3. Phase 1 — Tải dataset (~30 phút/session)

Nguồn: `https://cicresearch.ca/CICDataset/MalDroid-2020/Dataset/APKs/`

- Tải 5 file zip theo category, giải nén vào `/content/data/apks/<category>/`
- Sinh `manifest.csv`: `sha256, category, label, path, size`
- `label = 0` nếu Benign, `1` nếu còn lại (bài toán nhị phân trước; đa lớp để sau)
- **Lưu `manifest.csv` xuống Drive** — đây là thứ duy nhất cần giữ giữa các session ở phase này

Chấp nhận tải lại mỗi session, nhưng lưu feature đã trích xuống Drive để không phải trích lại.

---

## 4. Phase 2 — Trích feature tĩnh (~1 giờ, chạy 1 lần)

### Chỉ trích đúng cái cần — không trích đủ 17k

| Tập | Số APK | Dùng để |
|---|---|---|
| Train (stratified subsample) | 6.000 | Fit model + vectorizer |
| Test 500 bản sạch | 500 | Mốc F1 sạch cho phép so Δ |
| Test 500 × 6 kỹ thuật | 3.000 | Phase 5 |
| **Tổng** | **9.500** | ~1,5s/APK ÷ 4 worker ≈ **1 giờ** |

Hơn 11k APK còn lại không bao giờ được dùng — đừng trích.

### Bốn nhóm feature ở pass đầu

| Nhóm | Nội dung | Nguồn Androguard | Chi phí | Dự đoán độ nhạy |
|---|---|---|---|---|
| **G1 Manifest** | permissions, intent actions, số activity/service/receiver, exported flags, min/target SDK | `APK` | <0,5s | Thấp |
| **G2 API calls** | tập Android framework API được gọi (binary + count) | `DalvikVMFormat` | ~1s | Trung bình |
| **G3 Opcode** | phân bố opcode + 2-gram, mức method rồi tổng hợp | `DalvikVMFormat` | ~1s | Thấp |
| **G4 Strings** | URL, IP, số điện thoại, chuỗi base64, entropy trung bình | `DalvikVMFormat` | ~1s | **Rất cao** |

**G5 (FCG cấu trúc) hoãn sang pass sau.** `Analysis.get_call_graph()` tốn 5–30s/APK, tức chiếm ~80% tổng thời gian nếu bật ngay từ đầu. Chỉ thêm G5 nếu Bảng B ở Phase 5 cho thấy G1–G4 không đủ tách bạch.

### Baseline đối chiếu — giữ CSV của CIC

Song song với feature Androguard, train thêm một model trên bộ CSV gốc của CIC. Không dùng được cho phần obfuscation, nhưng đây là bộ mà phần lớn paper trên CICMalDroid sử dụng, nên con số sạch của bạn **so sánh được trực tiếp với literature**. Báo cáo cả hai.

JSON của CIC có trường ghi nhận dấu hiệu obfuscation trong mẫu gốc — dùng cho Bảng C (kiểm tra shortcut learning).

### Yêu cầu kỹ thuật

- Multiprocessing với `cpu_count()` worker, timeout **60s/APK** (Androguard treo trên APK hỏng)
- Checkpoint mỗi 500 APK → `features_part_NNN.parquet` trên Drive
- Ghi log riêng các APK fail (`failed.csv`) kèm lý do — **báo cáo tỉ lệ fail trong paper**
- Output cuối: `features_clean.parquet`, index theo sha256

---

## 5. Phase 3 — Baseline ML + eval sạch (1 giờ)

### Split
- Stratified theo `category`, tỉ lệ 60/20/20 train/val/test
- **Chốt cứng `test_sha256.txt` ngay tại đây và không bao giờ đổi.** Mọi so sánh sau đều trên đúng tập này.

### Model
Train 3 model, không cần tuning sâu — đây là baseline để so, không phải để khoe:
- Random Forest (n=300)
- XGBoost
- Linear SVM

Vectorize: fit vectorizer **chỉ trên train**, transform val/test. Sai ở đây là leakage.

### Báo cáo
Macro-F1, per-class F1, FPR trên lớp Benign, confusion matrix. Lưu `results/clean_baseline.json`.

Thêm một bước quan trọng: **train thử từng nhóm feature riêng lẻ (G1 một mình, G2 một mình...)**. Cần con số này để ở Phase 5 biết nhóm nào đang gánh model.

---

## 6. Phase 4 — Pipeline obfuscation (9–10 giờ, chia 2 session)

### Subsample
Từ `test_sha256.txt` lấy **500 APK stratified**: 250 benign + 250 malware (chia đều 4 category malware).

### Sáu nhóm kỹ thuật Obfuscapk

| Nhóm | Kỹ thuật | Tấn công vào |
|---|---|---|
| **T1 Trivial** | `Rebuild`, `NewAlignment`, `NewSignature` | Kiểm chứng: nếu T1 làm tụt accuracy thì model đang học artifact đóng gói, không phải hành vi |
| **T2 Rename** | `ClassRename`, `MethodRename`, `FieldRename` | G2, G4 |
| **T3 String encrypt** | `ConstStringEncryption`, `ResStringEncryption` | G4 |
| **T4 Asset/Lib encrypt** | `AssetEncryption`, `LibEncryption` | G4 |
| **T5 Control flow** | `Goto`, `ArithmeticBranch`, `Nop`, `Reorder` | G3, G5 |
| **T6 Reflection** | `CallIndirection`, `Reflection`, `AdvancedReflection` | G2, G5 |

### Ba quy tắc không được vi phạm

1. **Chỉ obfuscate tập test. Train giữ nguyên bản sạch.** Nếu obfuscate cả hai thì đang đo chuyện khác.
2. **Obfuscate cả benign lẫn malware.** Nếu chỉ obfuscate malware, model sẽ học luật tắt "obfuscate = độc" và ra số đẹp giả tạo.
3. **Verify APK sau obfuscate.** Loại khỏi tập đánh giá nếu `apksigner verify` fail hoặc Androguard không parse được. Ghi lại tỉ lệ hỏng theo từng kỹ thuật — bản thân nó là một kết quả đáng báo cáo.

Output: `data/apks_obf/<technique>/<sha256>.apk`, checkpoint tiến độ vào `obf_progress.json` trên Drive để resume sau timeout.

Sau đó chạy lại Phase 2 trên các APK đã obfuscate → `features_obf_<technique>.parquet`.

---

## 7. Phase 5 — Ma trận kết quả (2 giờ)

Đây là **contribution chính**. Không phải một con số tổng, mà một bảng.

### Bảng A — Mức tụt macro-F1 theo kỹ thuật

| Kỹ thuật | F1 sạch | F1 sau obf | Δ | FPR benign sau obf | % APK hỏng |
|---|---|---|---|---|---|
| T1 … T6 | | | | | |

### Bảng B — Ma trận kỹ thuật × nhóm feature

Train model chỉ-một-nhóm ở Phase 3, rồi test từng nhóm dưới từng kỹ thuật.

|  | G1 Manifest | G2 API | G3 Opcode | G4 String | G5 FCG |
|---|---|---|---|---|---|
| T1 Trivial | | | | | |
| T2 Rename | | | | | |
| T3 String | | | | | |
| T4 Asset | | | | | |
| T5 CFG | | | | | |
| T6 Reflection | | | | | |

Mỗi ô là ΔF1. **Bảng này chính là thứ chỉ ra phải sửa gì** — nhóm feature nào bất biến thì giữ, nhóm nào sụp thì thay bằng biểu diễn khác.

### Bảng C — Kiểm tra shortcut learning
Train một classifier chỉ để phân biệt "đã obfuscate / chưa obfuscate", bỏ qua nhãn malware. Nếu nó đạt F1 cao, feature space của bạn đang mã hóa mạnh dấu vết obfuscation → mọi kết quả detection đều đáng nghi.

---

## 8. Ghi chú cho Claude Code

- **Ưu tiên sửa nhỏ, tập trung.** Mỗi phase là một module độc lập, không refactor chéo.
- Mọi bước dài **phải resume được**. Giả định session Colab chết bất cứ lúc nào.
- Không cache ngầm: mỗi artifact có đường dẫn tường minh trên Drive.
- Đặt `RANDOM_SEED = 42` toàn cục, log seed vào mọi file kết quả.
- Không tuning hyperparameter để đẩy số ở Phase 3. Baseline là baseline.
- Khi số liệu trông quá đẹp (F1 > 0.99 dưới obfuscation), **nghi ngờ trước khi mừng** — kiểm tra Bảng C và kiểm tra APK có thật sự bị obfuscate không.

---

## 9. Sau khi xong (không nằm trong scope này)

- Trục thời gian: chuyển sang AndroZoo lọc theo `dex_date`, hoặc dùng benchmark LAMDA
- Nếu G4 (string) sụp mạnh như dự đoán → thử LLM sinh mô tả hành vi chuẩn hóa làm node feature cho FCG
- Đối chiếu với ELSA-RAMD benchmark để báo cáo theo chuẩn cộng đồng
