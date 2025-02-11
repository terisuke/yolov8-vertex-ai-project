# ベースイメージを指定 (NVIDIA CUDAを含むイメージ)
FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu20.04

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  git \
  && rm -rf /var/lib/apt/lists/*

# Ultralytics YOLOv8をインストール
RUN pip3 install ultralytics

# (オプション) その他の必要なライブラリをインストール (例: pandas, matplotlib)
# RUN pip3 install pandas matplotlib

# (オプション) トレーニングコードをコピー (後で説明)
# COPY train.py .

# (オプション) エントリーポイントを設定 (後で説明)
# ENTRYPOINT ["python3", "train.py"]