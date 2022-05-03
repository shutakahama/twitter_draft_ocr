import cv2
from PIL import Image
from pprint import pprint


def split_video(video_path, img_folder, interval_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    # 何frameごとにcaptureするか
    interval = int(cap.get(cv2.CAP_PROP_FPS) * interval_sec)

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
            cv2.imwrite(f"{img_folder}/{filled_second}.png", frame)
        idx += 1


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


def append_string(draft_list, new_str, min_length=10, search_num=10):
    # new_strがあまりに短すぎる場合は追加しない
    if len(new_str) < min_length:
        return draft_list
    # 一致判定の精度を上げるため空白は除く
    new_str = new_str.replace(" ", "")

    # 短い方(s2)が長い方(s1)のprefixまたはsuffixに一致するかを判定する
    def check_duplicate(s1, s2):
        return (levenshtein(s1[:len(s2)], s2) <= 0.5 * len(s2) or
                levenshtein(s1[-len(s2):], s2) <= 0.5 * len(s2))

    str_to_remove = []
    # draft_listの末尾search_num個を調査
    for s in draft_list[-search_num:]:
        # もしnew_strと一致してより長い文章が既にdraft_listにある場合、
        # new_strは追加するのを止める
        if len(s) >= len(new_str) and check_duplicate(s, new_str):
            return draft_list
        # もしnew_strと一致してより短い文章が既にdraft_listにある場合、
        # new_strを採用してdraft_listにある方を削除する
        elif len(s) < len(new_str) and check_duplicate(new_str, s):
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
        # 直前に認識した文字列の下端（prev_pos）と、今回の文字列の上端が
        # margin_ratio * line_heightより大きかった場合、別ツイートと見做す
        # append_stringを実行し、current_draftを初期化
        if prev_pos + margin_ratio * line_height <= box.position[0][1]:
            draft_list = append_string(draft_list, current_draft)
            current_draft = ''

        current_draft = current_draft + box.content
        prev_pos = box.position[1][1]

    draft_list = append_string(draft_list, current_draft)

    return draft_list


def read_strings_google_api(img_path, draft_list, margin_ratio):
    from google.cloud import vision
    import io

    client = vision.ImageAnnotatorClient()

    with io.open(img_path, "rb") as image_file:
        content = image_file.read()

    img = vision.Image(content=content)
    response = client.document_text_detection(image=img)

    # レスポンスからテキストデータを抽出
    for page in response.full_text_annotation.pages:
        # blocks内のblockは大抵1つのみ
        for block in page.blocks:
            prev_pos = -100
            current_draft = ''
            # paragraphは段落または行ごとに切れていることが多い
            # （tweet単位にはなっていない）
            for paragraph in block.paragraphs:
                bbox = paragraph.words[0].bounding_box.vertices
                upper = min([b.y for b in bbox])
                lower = max([b.y for b in bbox])
                line_height = lower - upper
                # 直前に認識した文字列の下端（prev_pos）と、今回の文字列の上端が
                # margin_ratio * line_heightより大きかった場合、別ツイートと見做す
                # append_stringを実行し、current_draftを初期化
                if prev_pos + margin_ratio * line_height <= upper:
                    draft_list = append_string(draft_list, current_draft)
                    current_draft = ''

                # wordsが含まれる単語のリストなので、それを逐次結合する
                # prev_posも逐次更新する
                for word in paragraph.words:
                    lower = max([b.y for b in word.bounding_box.vertices])
                    prev_pos = max(prev_pos, lower)
                    current_draft += ''.join([
                        symbol.text for symbol in word.symbols
                    ])

            draft_list = append_string(draft_list, current_draft)

    return draft_list
