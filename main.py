import argparse
import cv2
import json
import os
from PIL import Image


def levenshtein(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,         # insertion
                           dp[i][j - 1] + 1,         # deletion
                           dp[i - 1][j - 1] + cost)  # replacement

    return dp[n][m]


def check_duplicate(s1, s2):
    # 一致しているかどうかの判定と、もし一致していたらより完全と思われる方の文を返す
    if len(s1) <= len(s2):
        s1, s2 = s2, s1

    # if s2 in s1:
    #     return True, s1
    for i in range(len(s1) - len(s2) + 1):
        if levenshtein(s1[i:i+len(s2)], s2) <= 0.5 * len(s2):
            return True, s1

    return False, ""


def append_string(draft_list, new_str):
    if len(new_str) < 10:
        return draft_list
    new_str = new_str.replace(" ", "")

    # 重複して記録してしまうのを防ぐため、既に読み取った文のうち直近10件について、
    # 一致していないかを比較する
    # 一致判定されたら、より完全と思われる文の方のみを残す
    str_to_remove = []
    for s in draft_list[-10:]:
        dup, rs = check_duplicate(new_str, s)
        if dup and rs == s:
            return draft_list
        elif dup and rs == new_str:
            str_to_remove.append(s)

    for s in str_to_remove:
        draft_list.remove(s)

    draft_list.append(new_str)
    return draft_list


def read_strings_tesseract(img_path, draft_list, margin_ratio=1.5):
    import pyocr
    import pyocr.builders
    import pyocr.tesseract

    tesseract = pyocr.tesseract

    # tessaractを使って行ごとに文字列を認識
    img = Image.open(img_path)
    res = tesseract.image_to_string(
        img,
        lang='jpn',
        builder=pyocr.builders.LineBoxBuilder()
    )

    prev_pos = -100
    current_draft = ''
    for i, box in enumerate(res):
        line_height = box.position[1][1] - box.position[0][1]
        # 直前に認識した文字列の下端と、今回の文字列の上端がmargin_ratio * line_heightより
        # 大きかった場合、別ツイートと見做す
        # そうでなければ直前の文字列と連結する
        if prev_pos + margin_ratio * line_height <= box.position[0][1]:
            draft_list = append_string(draft_list, current_draft)
            current_draft = box.content
        else:
            current_draft = current_draft + box.content
        prev_pos = box.position[1][1]

    draft_list = append_string(draft_list, current_draft)

    return draft_list


def read_strings_google_api(img_path, draft_list):
    from google.cloud import vision
    import io

    client = vision.ImageAnnotatorClient()

    with io.open(img_path, "rb") as image_file:
        content = image_file.read()

    img = vision.Image(content=content)
    response = client.document_text_detection(image=img)

    # レスポンスからテキストデータを抽出
    for page in response.full_text_annotation.pages:
        blocks = page.blocks
        blocks.sort(key=lambda x: x.bounding_box.vertices[0].y)
        for bi, block in enumerate(blocks):
            # 両端のブロックは文章が見切れている恐れがあるので除く
            if bi == 0 or bi == len(blocks) - 1:
                continue
            current_draft = ''
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    current_draft += ''.join([
                        symbol.text for symbol in word.symbols
                    ])
            draft_list = append_string(draft_list, current_draft)

    return draft_list


def split_movie(video_path, img_folder, interval_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    interval = int(cap.get(cv2.CAP_PROP_FPS) * interval_sec)  # 何frameごとにcaptureするか

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % interval == 0:
            # 画像の上端と下端は不要な情報が混ざっている可能性が高いので除去する
            height = frame.shape[0]
            frame = frame[int(height * 1/8):int(height * 7/8)]
            filled_second = str(idx // interval).zfill(4)
            cv2.imwrite("{}/{}.png".format(img_folder, filled_second), frame)
        idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--margin-ratio', type=float, default=1.5)
    parser.add_argument('--frame-interval-sec', type=float, default=1.0)
    parser.add_argument('--ocr-mode', type=str, default="google_api")
    options, _ = parser.parse_known_args()
    return options


def main():
    options = parse_args()

    # 動画を読み込んで保存する
    img_folder = './cropped_images'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    split_movie(options.video_path, img_folder, options.frame_interval_sec)

    # 読み込む画像をリストアップする
    img_list = sorted([os.path.join(img_folder, fn) for fn in os.listdir(img_folder)])
    if options.reverse:
        img_list = img_list[::-1]
    # img_list = ['./draft3.png']

    # draft_listに認識した文字列を追加していく
    draft_list = []
    for i, img in enumerate(img_list):
        print(f"reading image {i+1}/{len(img_list)}")
        if options.ocr_mode == 'tesseract':
            draft_list = read_strings_tesseract(img, draft_list, options.margin_ratio)
        elif options.ocr_mode == 'google_api':
            draft_list = read_strings_google_api(img, draft_list)

    # 認識した結果を書き出す
    draft_json = {i: e for i, e in enumerate(draft_list)}
    with open('draft.json', 'w') as f:
        json.dump(draft_json, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
