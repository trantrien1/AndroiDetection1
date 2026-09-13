# apk-obfuscation-robustness

Đo độ bền của static Android malware detection dưới obfuscation, trên CICMalDroid 2020.

Mục tiêu không phải đẩy accuracy lên cao hơn, mà **đo xem accuracy hiện tại sụp đổ ở đâu và vì sao**. Kết quả không phải một con số mà là ba bảng (xem [PLAN.md](PLAN.md) mục 7).

## Chạy

Trên Colab: mở [notebooks/colab_driver.ipynb](notebooks/colab_driver.ipynb) và chạy tuần tự theo phase.

Từ dòng lệnh:

```bash
export APKROB_WORK=/content/drive/MyDrive/apk-robustness   # bền vững, sống qua session
export APKROB_SCRATCH=/content/apkrob                       # tạm, bị xoá mỗi session

python -m src.download --zip-dir ~/maldroid_zips   # Phase 1: giải nén + manifest.csv
python -m src.split                       # chốt cứng test_sha256.txt
python -m src.obfuscate --smoke-test      # Phase 0: KHÔNG đi tiếp nếu fail
python -m src.features.extract --tag clean        # Phase 2
python -m src.train                               # Phase 3
python -m src.evaluate --val --clean
python -m src.obfuscate --techniques T1_trivial T2_rename T3_string   # Phase 4, session A
python -m src.obfuscate --techniques T4_asset T5_cfg T6_reflection    # Phase 4, session B
for t in T1_trivial T2_rename T3_string T4_asset T5_cfg T6_reflection; do
  python -m src.features.extract --tag $t
done
python -m src.evaluate --obf              # Phase 5
python -m src.matrix                      # -> results/tables.md
```

Kiểm tra đường ống ML mà không cần APK (chạy trong ~2 phút):

```bash
python tests/smoke_synthetic.py
```

## Lấy dataset — không tải tự động được nữa

**URL trong PLAN mục 3 đã chết.** `https://cicresearch.ca/CICDataset/MalDroid-2020/Dataset/APKs/` giờ trả 302 về trang giới thiệu datasets của UNB, và CIC đã đặt toàn bộ dataset sau một **form đăng ký** (họ tên, email, tổ chức, chức danh, quốc gia). Không còn đường tải ẩn danh.

1. Mở https://www.unb.ca/cic/datasets/maldroid-2020.html → **Download the dataset** → điền form.
2. Tải 5 file zip theo category về một thư mục (nên để trên Drive để khỏi làm lại mỗi session).
3. `python -m src.download --zip-dir /đường/dẫn/tới/thư/mục/zip`

Tên file không cần khớp chính xác — miễn có chứa tên category, khớp không phân biệt hoa thường. Nếu bạn đã tự giải nén sẵn thành `apks/<category>/*.apk` thì chỉ cần `python -m src.download --manifest-only`.

`download.py` kiểm tra magic byte của mọi file zip trước khi giải nén và nhận diện riêng trang form của CIC, nên nếu CIC đổi gì nữa bạn sẽ thấy đúng nguyên nhân thay vì một `BadZipFile` khó hiểu ở tận bước sau.

### Hai con số dễ lẫn

Bộ **APK** có 17.341 mẫu — đúng như PLAN ghi. Nhưng bộ **đặc trưng CSV** (phân tích động, thứ phần lớn paper dùng) chỉ có 11.598 mẫu sau khi CIC loại các JSON hỏng, trong đó Benign là 1.795.

Hai con số này áp cho hai thứ khác nhau, và dự án này dùng cả hai: pipeline chính chạy trên APK (17.341), còn `--csv-baseline` chạy trên CSV (11.598) để đối chiếu với literature. Đừng so trực tiếp macro-F1 của hai bên như thể cùng một tập mẫu.

PLAN ghi benign ≈ 4.039; con số này chưa xác nhận được cho bộ APK. Hãy đọc `manifest.csv` và `splits/split_summary.json` làm số thật — nếu benign ít hơn 250 thì `src.split` sẽ cảnh báo và tập test không cân bằng được, lúc đó hạ `--n-test`.

## Cài đặt

Obfuscapk gọi ba công cụ qua PATH: **apktool**, **apksigner**, **zipalign**. Thiếu `apktool` thì mọi kỹ thuật obfuscation đều fail ngay ở bước decompile.

```bash
apt-get install -y openjdk-17-jdk-headless apktool zipalign apksigner

# Obfuscapk: KHÔNG pip install được — xem giải thích bên dưới
git clone --depth 1 https://github.com/ClaudiuGeorgiu/Obfuscapk.git /content/Obfuscapk
export OBFUSCAPK_SRC=/content/Obfuscapk/src

pip install -r requirements.txt
python -m src.obfuscate --smoke-test    # xác nhận cả chuỗi trên chạy được
```

### Hai cái bẫy khi cài Obfuscapk

`pip install obfuscapk` trong PLAN mục 2 **không bao giờ chạy được**, và cái thứ hai đợi ngay sau khi bạn vòng qua được cái thứ nhất:

1. **Obfuscapk không có trên PyPI.** `pip install obfuscapk` trả về `No matching distribution found`. Repo cũng không có `setup.py` ở gốc (nó nằm trong `src/`), nên `pip install git+...` cũng không được. Cách duy nhất là clone rồi đưa `src/` vào `PYTHONPATH` — đặt `OBFUSCAPK_SRC` và [src/obfuscate.py](src/obfuscate.py) tự lo phần còn lại cho mọi subprocess.

2. **Yapsy 1.12.2 hỏng trên Python 3.12.** `src/requirements.txt` của Obfuscapk ghim `Yapsy==1.12.2` — bản mới nhất trên PyPI, phát hành tháng 7/2019 — và bản đó `import imp`, module đã bị xoá khỏi Python 3.12 ([yapsy#19](https://github.com/tibonihoo/yapsy/issues/19)). Colab (Ubuntu 24.04) chạy đúng phiên bản này. Nhánh master của Yapsy đã chuyển sang `importlib` nhưng chưa bao giờ được phát hành, nên `requirements.txt` ở đây lấy Yapsy từ git.

Các pin còn lại của Obfuscapk (`pycryptodome==3.12.0`, 2021) không có wheel cho Python 3.12; `requirements.txt` ở đây cài bản mới nhất thay vì theo pin. Obfuscapk **không** phụ thuộc androguard nên không xung đột với `androguard==4.1.2`.

`python -m src.obfuscate --smoke-test` (hoặc `check_toolchain()`) nhận diện cả hai lỗi trên và in ra đúng lệnh cần chạy, thay vì để bạn phát hiện sau khi đã tải xong 17k APK.

## Cấu trúc

| Đường dẫn | Vai trò |
|---|---|
| [src/config.py](src/config.py) | Mọi đường dẫn + `RANDOM_SEED = 42` + định nghĩa 6 nhóm kỹ thuật |
| [src/download.py](src/download.py) | Phase 1 — tải dataset, sinh `manifest.csv` |
| [src/split.py](src/split.py) | Chốt cứng `test_sha256.txt` và tập con 250+250 để obfuscate |
| [src/features/groups.py](src/features/groups.py) | Định nghĩa 5 nhóm feature G1–G5 |
| [src/features/extract.py](src/features/extract.py) | Phase 2 — Androguard → feature dict, checkpoint mỗi 500 APK |
| [src/features/vectorize.py](src/features/vectorize.py) | dict → sparse matrix, fit **chỉ trên train**, cắt cột theo nhóm |
| [src/obfuscate.py](src/obfuscate.py) | Phase 4 — wrapper Obfuscapk song song, resume được |
| [src/train.py](src/train.py) | Phase 3 — RF/XGBoost/LinearSVM + model chỉ-một-nhóm |
| [src/evaluate.py](src/evaluate.py) | Metrics, eval sạch và eval obfuscated (so sánh ghép cặp) |
| [src/matrix.py](src/matrix.py) | Phase 5 — sinh Bảng A/B/C + cảnh báo tự động |

## Artifact sinh ra

Trên `$APKROB_WORK`:

```
manifest.csv
splits/    test_sha256.txt (đã chốt), train/val/obf/extract_sha256.txt, split_summary.json
features/  features_clean.parquet, features_obf_<T>.parquet, failed_<tag>.csv, parts/
models/    vectorizer.joblib, ALL__rf.joblib, G1__rf.joblib, ..., models_index.json
results/   clean_baseline.json, csv_baseline.json, obf_<T>.json,
           obf_broken_rate.csv, table_a.csv, table_b_delta.csv, tables.md, tables.json
obf_progress.json
```

## Hai chỗ PLAN.md tự mâu thuẫn, và cách giải

1. **Tập val.** Mục 4 cấp ngân sách trích feature cho 6.000 train + 500 test, không nhắc val; mục 5 đòi split 60/20/20. Vì mục 5 đã cấm tuning hyperparameter, val chỉ còn vai trò sanity-check, nên mặc định là **6.000 / 1.000 / 500** (7.500 clean). Đổi bằng `--n-train / --n-val / --n-test`.

2. **Thành phần tập test.** Mục 5 nói test là stratified theo category; mục 6 đòi tập obfuscate là 250 benign + 250 malware lấy *từ* test. Benign chỉ chiếm ~23% dataset nên một tập test 500 stratified chỉ có ~116 benign — không bao giờ đủ 250. Cách đọc duy nhất nhất quán: 500 APK ở phía test **chính là** 500 APK sẽ bị obfuscate, cân bằng 250/250. Ngân sách khi đó khớp đúng mục 4: 7.500 clean + 3.000 obfuscated. Dùng `--test-stratified` để quay về tỉ lệ tự nhiên, đổi lại tập obfuscate mất cân bằng.

Phụ phẩm của (2): 250 benign cho ước lượng FPR ổn hơn hẳn 116, mà FPR trên lớp Benign là một trong những số chính của Bảng A.

## Ba điều dễ làm sai

1. **Vectorizer fit lại trên tập obfuscated.** Nó sẽ học được cả token do obfuscator sinh ra, và mọi con số Δ mất nghĩa. Vectorizer chỉ fit một lần trên train sạch, sau đó đóng băng.
2. **So F1 sạch trên 500 APK với F1 obf trên 430 APK còn lại.** Mỗi kỹ thuật làm hỏng một số APK khác nhau. [src/evaluate.py](src/evaluate.py) tính lại F1 sạch trên đúng tập sống sót của từng kỹ thuật.
3. **Đọc Δ mà bỏ qua hàng T1.** T1 là control: nó đo ảnh hưởng của việc đóng gói lại, không phải của obfuscation. Phần đáng quan tâm ở T2–T6 là phần *vượt quá* mức tụt của T1.

## Trạng thái

Đường ống ML đã được kiểm chứng end-to-end bằng [tests/smoke_synthetic.py](tests/smoke_synthetic.py). Ba module chạm vào APK thật — `download.py`, `features/extract.py` (Androguard), `obfuscate.py` (Obfuscapk) — chưa chạy trên dữ liệu thật; smoke test ở Phase 0 và cell `--limit 20` ở Phase 2 là hai chốt kiểm tra chúng trước khi chạy dài.
