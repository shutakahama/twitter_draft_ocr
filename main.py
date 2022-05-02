import argparse
import json
import os
import time

from functions import (
    split_movie,
    read_strings_tesseract,
    read_strings_google_api
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--margin-ratio', type=float, default=1.5)
    parser.add_argument('--frame-interval-sec', type=float, default=1.0)
    parser.add_argument('--ocr-mode', type=str, default="google_api")
    parser.add_argument('--output-file', type=str, default="draft.json")
    options, _ = parser.parse_known_args()
    return options


def main():
    options = parse_args()
    start_time = time.time()

    # 動画を読み込んで保存する
    img_folder = './cropped_images'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    split_movie(options.video_path, img_folder, options.frame_interval_sec)

    # 読み込む画像をリストアップする
    img_list = sorted([os.path.join(img_folder, fn) for fn in os.listdir(img_folder)])

    # draft_listに認識した文字列を追加していく
    draft_list = []
    for i, img in enumerate(img_list):
        print(f"reading image {i+1}/{len(img_list)}")
        if options.ocr_mode == 'tesseract':
            draft_list = read_strings_tesseract(img, draft_list, options.margin_ratio)
        elif options.ocr_mode == 'google_api':
            draft_list = read_strings_google_api(img, draft_list, options.margin_ratio)

    # 認識した結果を書き出す
    draft_json = {i: e for i, e in enumerate(draft_list)}
    with open(options.output_file, 'w') as f:
        json.dump(draft_json, f, indent=4, ensure_ascii=False)

    print("time: {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
