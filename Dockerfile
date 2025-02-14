# マルチアーキテクチャビルドに対応するため、buildxを使用することを前提とします。
# buildx がインストールされていない場合は、`docker buildx install` でインストールしてください。

# builder を作成 (まだ作成していない場合)
# docker buildx create --use --name mybuilder

# builder を使用する
# docker buildx use mybuilder

# ベースイメージを指定 (NVIDIA CUDAを含むイメージ)
# Vertex AI Training では、特定のNVIDIAコンテナイメージを使用する必要があります。
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
# トレーニングでGPUを使用する場合は、以下から適切なイメージを選択してください。
# 例: FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04  # CUDA 11.8, cuDNN 8
# CPUのみでトレーニングする場合は、より軽量なイメージを選択できます。
# 例: FROM python:3.9-slim-buster

# GPU を使用する場合 (例)
# 可能であれば runtime イメージを試す (ただし、ビルドに失敗する可能性あり)
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# CPU のみを使用する場合 (例)
# FROM python:3.9-slim-buster

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  git \
  libgl1-mesa-dev \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# requirements.txt をコピー (先にコピーすることで、変更がない限りキャッシュが効く)
COPY requirements.txt /app/

# 依存関係のインストール (requirements.txt を使用)
# RUN pip3 install --no-cache-dir -r requirements.txt　修正前
RUN pip3 install --no-cache-dir -r requirements.txt

# app.py と utils ディレクトリをコピー
COPY app.py /app/
COPY utils /app/utils
COPY data.yaml /app/


# サービスアカウントキーをコピー (Vertex AI でサービスアカウントを使う場合)
# サービスアカウントキーを安全に扱うように注意してください。
# 可能であれば、Workload Identity 連携の使用を検討してください。
COPY service-account-key.json /app/

# 環境変数を設定
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json
ENV APP_ROOT=/app

# エントリーポイントを設定 (Vertex AI Custom Training で実行するコマンド)
ENTRYPOINT ["python3", "/app/app.py"]