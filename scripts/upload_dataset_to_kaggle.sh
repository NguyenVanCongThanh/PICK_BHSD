#!/bin/bash
# ==============================================================================
# Script tự động đẩy Dataset preprocessed_bhsd_pick (20GB) lên Kaggle Dataset
# ==============================================================================

DATA_DIR="/home/u001015/notebook/preprocessed_bhsd_pick"
KAGGLE_CONFIG="$HOME/.kaggle/kaggle.json"

echo "=== [1/4] Kiểm tra Kaggle API Token ==="
if [ ! -f "$KAGGLE_CONFIG" ]; then
    echo "[!] Chưa tìm thấy file $KAGGLE_CONFIG"
    echo "Hướng dẫn:"
    echo "  1. Đăng nhập vào https://www.kaggle.com"
    echo "  2. Vào Avatar góc phải -> Settings -> cuộn xuống mục 'API' -> bấm 'Create New Token'"
    echo "  3. Trình duyệt sẽ tải về file 'kaggle.json'"
    echo "  4. Mở file đó, copy nội dung và dán vào máy chủ bằng lệnh:"
    echo "       mkdir -p ~/.kaggle"
    echo "       nano ~/.kaggle/kaggle.json   # hoặc dán nội dung vào đây"
    echo "       chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

chmod 600 "$KAGGLE_CONFIG"
echo "[OK] Đã tìm thấy Kaggle Token."

echo "=== [2/4] Kiểm tra thư viện kaggle CLI ==="
if ! command -v kaggle &> /dev/null; then
    echo "[*] Đang cài đặt kaggle CLI..."
    python3 -m pip install --user kaggle
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== [3/4] Khởi tạo Metadata cho Kaggle Dataset ==="
cd "$DATA_DIR"

# Lấy username từ kaggle.json
KAGGLE_USER=$(grep -o '"username": *"[^"]*"' "$KAGGLE_CONFIG" | cut -d'"' -f4)

if [ -z "$KAGGLE_USER" ]; then
    echo "[!] Không đọc được username từ $KAGGLE_CONFIG. Vui lòng kiểm tra lại file json."
    exit 1
fi

echo "[*] Kaggle Username: $KAGGLE_USER"

cat <<EOF > "$DATA_DIR/dataset-metadata.json"
{
  "title": "preprocessed-bhsd-pick",
  "id": "${KAGGLE_USER}/preprocessed-bhsd-pick",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
EOF

echo "[OK] Đã tạo file dataset-metadata.json:"
cat "$DATA_DIR/dataset-metadata.json"

echo ""
echo "=== [4/4] Bắt đầu upload dataset lên Kaggle ==="
echo "[*] Đang tải ~20GB dữ liệu trực tiếp từ máy chủ lên Kaggle. Vui lòng đợi trong ít phút..."
kaggle datasets create -p "$DATA_DIR" -r zip --public

echo ""
echo "=== HOÀN TẤT ==="
echo "Dataset đã được tải lên: https://www.kaggle.com/datasets/${KAGGLE_USER}/preprocessed-bhsd-pick"
echo "Bây giờ bạn có thể vào Kaggle Notebook, bấm '+ Add Data' và tìm 'preprocessed-bhsd-pick'."
