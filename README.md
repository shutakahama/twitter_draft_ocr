# Twitter Draft OCR

スマホなどに保存しているTwitterの下書きをテキストデータとして取り出せる便利ツールです。  
詳細はこちらの記事をご覧ください。  

TBD

## 使い方
まず、Twitterの下書き画面をスクロールした動画を撮影して保存してください。  

OCRエンジンとしては、tesseractとGoogle Vision APIの2つを用意しています。

### tesseractを使う場合
```
pip install -r requirements.txt
python main.py --video-path YOUR_VIDEO_PATH --frame-interval-sec 0.2 --margin-ratio 1.5 --ocr-mode tesseract
```

### Google Vision APIを使う場合
予めGoogle Vision APIの設定をしておく必要があります。  
https://cloud.google.com/vision/docs/setup

```
pip install -r requirements.txt
python main.py --video-path YOUR_VIDEO_PATH --frame-interval-sec 0.2 --margin-ratio 1.5 --ocr-mode google_api
```