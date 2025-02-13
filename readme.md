```
gsutil -m cp -r datasets/house/* gs://yolo-v8-training/house/
```

ビルドとプッシュ、成功おめでとうございます！長い道のりでしたが、これで Vertex AI でのトレーニングの準備が整いました。

次のアクションは、Vertex AI Custom Training ジョブを作成して、トレーニングを実行することです。

手順:

Google Cloud Console で Vertex AI のページに移動:

Google Cloud Console (https://console.cloud.google.com/) にアクセスし、プロジェクト yolov8environment を選択します。

ナビゲーションメニュー (左上のハンバーガーアイコン) から、「Vertex AI」>「トレーニング」を選択します。

「トレーニング パイプライン」で「作成」をクリック:

「トレーニング パイプラインを作成」画面が表示されます。

トレーニング方法の選択:

「トレーニング方法」で「カスタム トレーニング（詳細オプション）」を選択します。

「続行」をクリック

モデルの詳細:

モデル名を入力します（例: yolov8-custom-model）。これはVertex AI上で管理するためのモデルの名前です。

「続行」をクリック。

トレーニング コンテナ:

「コンテナの選択」で、「カスタムコンテナ」を選択します。

「コンテナイメージ」に、先ほど Artifact Registry にプッシュしたイメージの URI を入力します。

asia-northeast1-docker.pkg.dev/yolov8environment/yolov8-repository/yolov8-training-image:v1

「続行」をクリック。

ハイパーパラメータ: (重要)

ここで、app.py に渡すコマンドライン引数を設定します。

「引数を追加」をクリックして、必要な引数を追加していきます。

例:

--epochs: 50 (エポック数)

--batch_size: 16 (バッチサイズ)

--model: yolov8n-seg.pt (使用するモデル。yolov8s-seg.pt なども選択可)

--imgsz: 640 (入力画像のサイズ)

--optimizer: Adam (オプティマイザ。SGD なども選択可)

--upload_bucket: yolo-v8-training (学習済みモデルをアップロードするバケット)

--upload_dir: trained_models (モデルをアップロードするバケット内のディレクトリ)

その他のハイパーパラメータ (必要に応じて)

引数の設定は、app.py の argparse の設定と対応している必要があります。

コンピューティングと料金:

「コンピューティング クラスタ」で、「新しいコンピューティング クラスタを作成」または既存のクラスタを選択します。

リージョン: asia-northeast1 (Artifact Registry と同じリージョン)

マシンタイプ:

GPU を使用する場合: n1-standard-8 (またはそれ以上) と、アクセラレータ タイプ NVIDIA Tesla T4 (または他の GPU) を選択。GPU の数も選択。

CPU のみを使用する場合: n1-standard-4 (またはそれ以上) を選択。

必要に応じて、ディスクサイズなどを調整します。

「クラスタを作成」をクリック (新しいクラスタを作成する場合)。

トレーニングジョブを開始:

サービスアカウント：以前作成したサービスアカウント(yolov8-training-saなど)を指定します.

設定内容を確認し、「トレーニングを開始」をクリックします。

補足:

料金: Vertex AI の利用には料金がかかります。特に GPU を使用する場合は、費用が高くなる可能性があります。トレーニングを開始する前に、料金体系を確認し、予算内で実行できることを確認してください。

トレーニング時間: トレーニング時間は、データセットのサイズ、モデルの複雑さ、エポック数、使用するハードウェア (GPU/CPU) によって大きく異なります。

ログ: トレーニングの進行状況は、Vertex AI の「トレーニング パイプライン」のページで確認できます。ログもここから確認できます。

Workload Identity 連携(より安全): 可能であれば、サービスアカウントキーをコンテナに含める代わりに、Workload Identity連携を使用することを強く推奨します。設定方法はやや複雑ですが、より安全にGCPリソースにアクセスできます。

これで、Vertex AI でカスタムトレーニングジョブが開始されます。トレーニングが完了すると、Cloud Storage の指定したバケットに学習済みモデル (best.pt) が保存されます。

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag asia-northeast1-docker.pkg.dev/yolov8environment/yolov8-repository/yolov8-training-image:v1 \
  --push .

[+] Building 4016.2s (26/26) FINISHED                                                           docker-container:mybuilder
 => [internal] load build definition from Dockerfile                                                                  0.0s
 => => transferring dockerfile: 2.34kB                                                                                0.0s
 => [linux/arm64 internal] load metadata for docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04                    2.6s
 => [linux/amd64 internal] load metadata for docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04                    2.7s
 => [auth] nvidia/cuda:pull token for registry-1.docker.io                                                            0.0s
 => [internal] load .dockerignore                                                                                     0.0s
 => => transferring context: 2B                                                                                       0.0s
 => [linux/amd64 1/9] FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04@sha256:28cb396884380adc15a4bda23e  253.8s
 => => resolve docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04@sha256:28cb396884380adc15a4bda23e9654610cbc4e20  0.0s
 => => sha256:b3b26ce36551048f3e8f5a3022a7a5f0051bc7417463524a2767ca9e002d295b 1.43GB / 1.43GB                      147.2s
 => => sha256:0723abde19c09431db80b5d0f0e003541f6232c2da5c4f35c635c861875937ae 85.88kB / 85.88kB                      0.4s
 => => sha256:d67c111fa58886d8904335b5122039816c8e51af5844cb6fe4e14154d9a7c016 2.45GB / 2.45GB                      217.8s
 => => sha256:464a8f74544589bf7b57f9a4cadcb6681e5ed00758f6c35025e691df4e88e890 1.52kB / 1.52kB                        0.4s
 => => sha256:dfecd7e9912b76ed460b8edd5a85f1943666e38a973ab5458177cf2c7c3110e3 1.68kB / 1.68kB                        0.3s
 => => sha256:ab17245097e491b9368790714f9d90ed447bf0973bd677cfe6f2456d62b72a13 62.28kB / 62.28kB                      0.5s
 => => sha256:8f6c9048534734f4c873935293b7296225846ceb31c1a158400a67ea170dde7f 1.38GB / 1.38GB                      148.6s
 => => sha256:95d7b781703928cf3c4eece39d800cccb76728c375fedf51ecd83833fb25e458 6.88kB / 6.88kB                        0.4s
 => => sha256:9fe6e2e61518cba6844870c03b285737daec35e62baf25ae7744629ed3a7b470 56.23MB / 56.23MB                     12.1s
 => => sha256:41f16248e682693ff20b3032c1d5e5541cc87c5af898ae2ff9b24d2940e59100 187B / 187B                            0.2s
 => => sha256:09d415c238d76b32a7ea4a6e6add9542db9a5641f7f183af70aae185d0709e58 7.94MB / 7.94MB                        2.7s
 => => sha256:96d54c3075c9eeaed5561fd620828fd6bb5d80ecae7cb25f9ba5f7d88ea6e15c 27.51MB / 27.51MB                      6.5s
 => => extracting sha256:96d54c3075c9eeaed5561fd620828fd6bb5d80ecae7cb25f9ba5f7d88ea6e15c                             0.5s
 => => extracting sha256:09d415c238d76b32a7ea4a6e6add9542db9a5641f7f183af70aae185d0709e58                             0.1s
 => => extracting sha256:9fe6e2e61518cba6844870c03b285737daec35e62baf25ae7744629ed3a7b470                             0.5s
 => => extracting sha256:41f16248e682693ff20b3032c1d5e5541cc87c5af898ae2ff9b24d2940e59100                             0.0s
 => => extracting sha256:95d7b781703928cf3c4eece39d800cccb76728c375fedf51ecd83833fb25e458                             0.0s
 => => extracting sha256:8f6c9048534734f4c873935293b7296225846ceb31c1a158400a67ea170dde7f                            13.2s
 => => extracting sha256:ab17245097e491b9368790714f9d90ed447bf0973bd677cfe6f2456d62b72a13                             0.0s
 => => extracting sha256:dfecd7e9912b76ed460b8edd5a85f1943666e38a973ab5458177cf2c7c3110e3                             0.0s
 => => extracting sha256:464a8f74544589bf7b57f9a4cadcb6681e5ed00758f6c35025e691df4e88e890                             0.0s
 => => extracting sha256:d67c111fa58886d8904335b5122039816c8e51af5844cb6fe4e14154d9a7c016                            21.9s
 => => extracting sha256:0723abde19c09431db80b5d0f0e003541f6232c2da5c4f35c635c861875937ae                             0.0s
 => => extracting sha256:b3b26ce36551048f3e8f5a3022a7a5f0051bc7417463524a2767ca9e002d295b                            14.0s
 => [linux/arm64 1/9] FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04@sha256:28cb396884380adc15a4bda23e  316.9s
 => => resolve docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04@sha256:28cb396884380adc15a4bda23e9654610cbc4e20  0.0s
 => => sha256:72ab55d23b189842c164c614c359d3f582c069e9ebcb84d2c5777d1981736420 1.44GB / 1.44GB                      163.4s
 => => sha256:c3d1aba441ed79183efc70233fcd21008ab935a7ca6fe53bfa3f629799c03968 84.56kB / 84.56kB                      1.0s
 => => sha256:b8492b90fe639d4d3a2bae7fe8b8fe9ffc7b18b769a6978a791cd3b5439b3f22 1.52kB / 1.52kB                        0.5s
 => => sha256:c1d4af1c3b59f52787d697a9c8cae85acc7c66f4348138e7d9ea636378566405 1.90GB / 1.90GB                      136.0s
 => => sha256:348eaaa9a0b9fa4ff7bdf8874f49b639d2ff1a383b8f293f9b6914da1b70451a 1.68kB / 1.68kB                        0.5s
 => => sha256:80f4ee570d79f723a491bcc4ce3673557b0896c8bbf141c6081643fc42d81d55 61.73kB / 61.73kB                      0.5s
 => => sha256:7430b86a779d7e38a28abfb018c2d42ae49eb518a025a4b2c92a077e00791d77 1.20GB / 1.20GB                      117.0s
 => => sha256:7a0e945d3737890a64837bb7add376469f383f265f26b87e5554940498315366 186B / 186B                            1.8s
 => => sha256:4ca3033696bd5d74ae1ae62a6f8c7571fe98f5a1503fcfa9ce038e040de6ff78 366.04kB / 366.04kB                    0.3s
 => => sha256:9b8e6ed38a81bab0a4bda6cebb8339d049da1578229ab3e46f9f9fc0e2149283 7.78MB / 7.78MB                        1.0s
 => => sha256:915eebb74587f0e5d3919cb77720c143be9a85a8d2d5cd44675d84c8c3a2b74a 25.97MB / 25.97MB                      5.5s
 => => extracting sha256:915eebb74587f0e5d3919cb77720c143be9a85a8d2d5cd44675d84c8c3a2b74a                             0.3s
 => => extracting sha256:9b8e6ed38a81bab0a4bda6cebb8339d049da1578229ab3e46f9f9fc0e2149283                             0.1s
 => => extracting sha256:4ca3033696bd5d74ae1ae62a6f8c7571fe98f5a1503fcfa9ce038e040de6ff78                             0.0s
 => => extracting sha256:7a0e945d3737890a64837bb7add376469f383f265f26b87e5554940498315366                             0.0s
 => => extracting sha256:95d7b781703928cf3c4eece39d800cccb76728c375fedf51ecd83833fb25e458                             0.0s
 => => extracting sha256:7430b86a779d7e38a28abfb018c2d42ae49eb518a025a4b2c92a077e00791d77                            10.6s
 => => extracting sha256:80f4ee570d79f723a491bcc4ce3673557b0896c8bbf141c6081643fc42d81d55                             0.0s
 => => extracting sha256:348eaaa9a0b9fa4ff7bdf8874f49b639d2ff1a383b8f293f9b6914da1b70451a                             0.0s
 => => extracting sha256:b8492b90fe639d4d3a2bae7fe8b8fe9ffc7b18b769a6978a791cd3b5439b3f22                             0.0s
 => => extracting sha256:c1d4af1c3b59f52787d697a9c8cae85acc7c66f4348138e7d9ea636378566405                            18.8s
 => => extracting sha256:c3d1aba441ed79183efc70233fcd21008ab935a7ca6fe53bfa3f629799c03968                             0.0s
 => => extracting sha256:72ab55d23b189842c164c614c359d3f582c069e9ebcb84d2c5777d1981736420                            13.3s
 => [internal] load build context                                                                                     0.0s
 => => transferring context: 7.89kB                                                                                   0.0s
 => [linux/amd64 2/9] WORKDIR /app                                                                                    0.5s
 => [linux/amd64 3/9] RUN apt-get update && apt-get install -y --no-install-recommends   python3   python3-pip   gi  47.0s
 => [linux/amd64 4/9] RUN pip3 install ultralytics                                                                  392.2s
 => [linux/arm64 2/9] WORKDIR /app                                                                                    0.6s
 => [linux/arm64 3/9] RUN apt-get update && apt-get install -y --no-install-recommends   python3   python3-pip   gi  23.8s
 => [linux/arm64 4/9] RUN pip3 install ultralytics                                                                   58.5s
 => [linux/arm64 5/9] RUN pip3 install google-cloud-storage                                                           3.9s
 => [linux/arm64 6/9] RUN pip3 install pyyaml                                                                         1.1s
 => [linux/arm64 7/9] COPY app.py /app/                                                                               0.0s
 => [linux/arm64 8/9] COPY utils /app/utils                                                                           0.0s
 => [linux/arm64 9/9] COPY service-account-key.json /app/                                                             0.0s
 => [linux/amd64 5/9] RUN pip3 install google-cloud-storage                                                           5.5s
 => [linux/amd64 6/9] RUN pip3 install pyyaml                                                                         1.2s
 => [linux/amd64 7/9] COPY app.py /app/                                                                               0.0s
 => [linux/amd64 8/9] COPY utils /app/utils                                                                           0.0s
 => [linux/amd64 9/9] COPY service-account-key.json /app/                                                             0.0s
 => exporting to image                                                                                             3313.1s
 => => exporting layers                                                                                             195.8s
 => => exporting manifest sha256:3e5c9547a8c267162a356224e42c77c2ee55bb1d81864050324b156564449cc7                     0.0s
 => => exporting config sha256:a7edf1c0b255b9d7d7bfc7c648aa7c07bbca460e8e9210530d393e64021df6c4                       0.0s
 => => exporting attestation manifest sha256:ec071f2a620b0fa97e41f4fbcd1653f30eaaf4769420d9649678787eb5357a53         0.0s
 => => exporting manifest sha256:03d6b7e9ed2adfa08226b865cf438dd77627af22dce2f85cf3b87affaf2702a7                     0.0s
 => => exporting config sha256:20c13fda7a29d64aa72705614f6c208ad0d06308ed71fbcf0d49d52c60a674d8                       0.0s
 => => exporting attestation manifest sha256:53228d11b69cb702e9930ccc7427ffcb3b03b8378e6d10957c0fb06fd0a31064         0.0s
 => => exporting manifest list sha256:6eb1dc8e12a97ba3c04dfeebcb545d8a849ba2ddc15d1c51ee2f3677d92701ae                0.0s
 => => pushing layers                                                                                              3113.7s
 => => pushing manifest for asia-northeast1-docker.pkg.dev/yolov8environment/yolov8-repository/yolov8-training-image  3.6s
 => [auth] yolov8environment/yolov8-repository/yolov8-training-image:pull,push token for asia-northeast1-docker.pkg.  0.0s

View build details: docker-desktop://dashboard/build/mybuilder/mybuilder0/wip4l15nwqrf8vgeaaumudd7p

 1 warning found (use docker --debug to expand):
 - SecretsUsedInArgOrEnv: Do not use ARG or ENV instructions for sensitive data (ENV "GOOGLE_APPLICATION_CREDENTIALS") (line 53)