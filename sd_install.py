# -*- coding: utf-8 -*-
import sys
import subprocess
from distutils.util import strtobool

# 引数解析
## 以下の辞書で値がTrueのオプションは、続けてその値を指定するもの
## 値がFalseのオプションは何らかのスイッチ（True／False値を持つ）
## 例：python3 kw_args.py -x hoge -y huga -z
options = {'-lm': True, '-db': True, '-cn': True}
args = {'lm': True, 'db': False, 'cn': True}  # デフォルト値

for key in options.keys():
    if key in sys.argv:
        idx = sys.argv.index(key)
        if options[key]:
            value = sys.argv[idx+1]
            if value.startswith('-'):
                raise ValueError(f'option {key} must have a value.')
            args[key[1:]] = value
            del sys.argv[idx:idx+2]
        else:
            args[key[1:]] = True
            del sys.argv[idx]

print("args['lm']:", args['lm'])
print("args['db']:", args['db'])
print("args['cn']:", args['cn'])

# 残ったsys.argvの要素は位置引数と考える（スクリプトファイル名を除く）
print(f'positional argument:', sys.argv[1:])

# ---------------------------------------------

# webuiをインストール
cmd = f'git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui /content/stable-diffusion-webui'
subprocess.call(cmd, shell=True)

# sd_dreambooth_extension のダウンロード
dreambooth = strtobool(args['db'])
if dreambooth:
  cmd = f'git clone https://github.com/d8ahazard/sd_dreambooth_extension.git /content/stable-diffusion-webui/extensions/sd_dreambooth_extension'
  subprocess.call(cmd, shell=True)

# モデルダウンロード関数の定義
cmd = f'apt-get -y install -qq aria2' # aria2を使うと爆速で、モデルをダウンロードできる
subprocess.call(cmd, shell=True)

def download_model(download_model_url, output_dir):
  # urlの最後がモデル名として、ファイル名を取得する
  file_name_path = download_model_url.split('/')[-1]
  # ダウンロードの実行
  cmd = f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {download_model_url} -d {output_dir} -o {file_name_path}'
  subprocess.call(cmd, shell=True)
  # ダウンロード後の配置されたディレクトリを一応確認
  cmd = f'ls -lh {output_dir}'
  subprocess.call(cmd, shell=True)

modelsDir = "/content/stable-diffusion-webui/models/Stable-diffusion"

# stable-diffsion v1.4 のモデルのダウンロード
download_model("https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt", modelsDir)

# v1.4以外の学習済みモデルをダウンロード
learnedModel = strtobool(args['lm'])    # True = ダウンロードする，False = ダウンロードしない

if learnedModel:
  # stable-diffsion v2.1 model(768×768)のダウンロード
  download_model("https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors", modelsDir)
  cmd = f'wget "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml" -O {modelsDir}/v2-1_768-ema-pruned.yaml'
  subprocess.call(cmd, shell=True)

  # stable-diffsion v2.1 base model(512×512)のダウンロード
  download_model("https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors", modelsDir)
  cmd = f'wget "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml" -O {modelsDir}/v2-1_512-ema-pruned.yaml'
  subprocess.call(cmd, shell=True)

  # VAE
  download_model("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors", "/content/stable-diffusion-webui/models/VAE")

  # dreamlike-art
  download_model("https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors", modelsDir)

  # basil_mix
  #download_model("https://huggingface.co/nuigurumi/basil_mix/resolve/main/Basil_mix_fixed.safetensors", modelsDir)

  # Realism Engine 1.0
  #!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://civitai.com/api/download/models/20414?type=Model&format=SafeTensor" -d {modelsDir} -o "realismEngine_v10.safetensors"
  #!wget "https://civitai.com/api/download/models/20414?type=Config&format=Other" -O {modelsDir}/realismEngine_v10.yaml

  # nfixer & nrealfixer for Illuminati Diffusion v1.1
  #!wget "https://civitai.com/api/download/models/15921" -O /content/stable-diffusion-webui/embeddings/nfixer.pt
  #!wget "https://civitai.com/api/download/models/15927" -O /content/stable-diffusion-webui/embeddings/nrealfixer.pt

  # ChilloutMix
  #!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://civitai.com/api/download/models/11745" -d {modelsDir} -o "chilloutmix_NiPrunedFp32Fix.safetensors"

# sd-webui-controlnetモデルをダウンロード
controlnet = strtobool(args['cn'])    # True = controlnetが動く，False = controlnetが動かない

controlnetModelsDir = "/content/stable-diffusion-webui/models/ControlNet"
if controlnet:
  # sd-webui-controlnet のダウンロード
  cmd = f'git clone https://github.com/Mikubill/sd-webui-controlnet /content/stable-diffusion-webui/extensions/sd-webui-controlnet'
  subprocess.call(cmd, shell=True)

  # ControlNet 1.1
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.yaml", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth", controlnetModelsDir)
  download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.yaml", controlnetModelsDir)

sys.exit()
