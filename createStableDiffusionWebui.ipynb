# コンテンツルートに移動
%cd /content

# webuiをインストール
!git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui

# sd-webui-controlnet のダウンロード
!git clone https://github.com/Mikubill/sd-webui-controlnet /content/stable-diffusion-webui/extensions/sd-webui-controlnet

# 作業ディレクトリを移動させておく
%cd /content/stable-diffusion-webui

# モデルダウンロード関数の定義
!apt-get -y install -qq aria2 # aria2を使うと爆速で、モデルをダウンロードできる
def download_model(download_model_url, output_dir):
  # urlの最後がモデル名として、ファイル名を取得する
  file_name_path = download_model_url.split('/')[-1]
  # ダウンロードの実行
  !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {download_model_url} -d {output_dir} -o {file_name_path}
  # ダウンロード後の配置されたディレクトリを一応確認
  !ls -lh {output_dir}

# stable-diffsion v1.4 のモデルのダウンロード
download_model("https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt", "/content/stable-diffusion-webui/models/Stable-diffusion")

# stable-diffsion v2.1 model(768×768)のダウンロード
#download_model("https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt", "/content/stable-diffusion-webui/models/Stable-diffusion")
#!wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -O /content/stable-diffusion-webui/models/Stable-diffusion/v2-1_768-ema-pruned.yaml

# stable-diffsion v2.1 base model(512×512)のダウンロード
#download_model("https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt", "/content/stable-diffusion-webui/models/Stable-diffusion")
#!wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml -O /content/stable-diffusion-webui/models/Stable-diffusion/v2-1_512-ema-pruned.yaml

# sd-webui-controlnetモデルをダウンロード
## この変数がTrue = controlnetが動く，False = controlnetが動かない
controlnet = True

if controlnet:
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")
  download_model("https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth", "/content/stable-diffusion-webui/extensions/sd-webui-controlnet/models")

# webuiを別URLで立ち上げる
!python launch.py --share --xformers --enable-insecure-extension-access
