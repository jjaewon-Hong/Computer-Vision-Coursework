"""
Report #2 - 구현 2
수식을 이용하여 BGR -> YCbCr, YCbCr -> BGR 변환을 직접 구현
(강의자료 For SDTV 수식 기준, full range 0~255 사용 - OpenCV cvtColor와 동일 범위)

[BGR -> YCbCr]
Y  = 0.299R + 0.587G + 0.114B
Cb = -0.172R - 0.339G + 0.511B + 128
Cr =  0.511R - 0.428G - 0.083B + 128

[YCbCr -> BGR]
R = Y + 1.371(Cr - 128)
G = Y - 0.698(Cr - 128) - 0.336(Cb - 128)
B = Y + 1.732(Cb - 128)
"""

import os           # 파일 경로 처리용 표준 라이브러리
import cv2 as cv     # 이미지 입출력(imread)과 화면 출력(imshow)에만 사용 (색공간 변환은 직접 구현)
import numpy as np   # 채널 분리, 행렬 연산(수식 계산), clip, dstack 등 배열 연산에 사용


def bgr2ycrcb_manual(bgr_img):
    """주어진 수식을 이용해 BGR -> YCbCr 변환을 직접 구현"""

    # 정수(uint8) 상태로 연산하면 음수/255 초과 값이 잘려나가는 오버플로우가 발생하므로
    # 실수형(float64)으로 변환 후 계산
    img = bgr_img.astype(np.float64)

    # OpenCV는 채널 순서가 B, G, R이므로 인덱스 0, 1, 2에서 각각 B, G, R을 추출
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # 강의자료 수식(For SDTV) 그대로 Y, Cb, Cr 계산
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.172 * R - 0.339 * G + 0.511 * B + 128
    Cr = 0.511 * R - 0.428 * G - 0.083 * B + 128

    # 계산 과정에서 0~255 범위를 벗어날 수 있으므로 클리핑(clipping)하여 유효 범위로 제한
    Y = np.clip(Y, 0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    return Y, Cb, Cr


def ycrcb2bgr_manual(Y, Cb, Cr):
    """주어진 수식을 이용해 YCbCr -> BGR 변환을 직접 구현"""

    # 입력 채널들을 실수형으로 변환 (정밀한 계산을 위해)
    Y = Y.astype(np.float64)
    Cb = Cb.astype(np.float64)
    Cr = Cr.astype(np.float64)

    # 강의자료 수식(For SDTV)의 역변환 그대로 R, G, B 계산
    R = Y + 1.371 * (Cr - 128)
    G = Y - 0.698 * (Cr - 128) - 0.336 * (Cb - 128)
    B = Y + 1.732 * (Cb - 128)

    # 계산 결과가 0~255 범위를 벗어날 수 있으므로 클리핑
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    # OpenCV 형식(B, G, R 순서)에 맞춰 채널을 쌓고, 화면 출력을 위해 uint8(8bit 정수)로 변환
    bgr = np.dstack([B, G, R]).astype(np.uint8)
    return bgr


def rgb_to_gray_to_rgb_manual(bgr_img):
    """수식을 이용해 BGR -> YCbCr -> Y채널(흑백) 추출 -> BGR 복원"""

    # 위에서 정의한 함수로 BGR -> YCbCr 변환 수행
    Y, Cb, Cr = bgr2ycrcb_manual(bgr_img)

    # Y채널만 추출한 흑백 영상 (디스플레이용, 8bit 정수형으로 변환)
    gray = Y.astype(np.uint8)

    # 복원: YCbCr -> BGR (Y, Cb, Cr 채널을 모두 이용한 완전한 역변환)
    restored_bgr = ycrcb2bgr_manual(Y, Cb, Cr)

    return gray, restored_bgr


# 이 파일을 직접 실행했을 때만 아래 코드가 동작 (다른 파일에서 import 될 때는 실행되지 않음)
if __name__ == '__main__':
    # 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡음
    imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    # 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드함
    img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

    # 위에서 정의한 함수를 실행하여 흑백(Y채널) 영상과 복원된 컬러 영상을 얻음
    gray_manual, restored_manual = rgb_to_gray_to_rgb_manual(img)

    # 원본, 흑백(Y채널), 복원된 영상을 각각 별도의 창으로 띄워서 비교
    # imshow만 호출하면 창 위치를 OS가 정하므로, namedWindow로 창을 먼저 만들고
    # moveWindow로 좌표를 직접 지정해서 원하는 순서대로 왼쪽부터 나란히 배치됨
    windows = [
        ('original', img),                   # 1번째: 원본
        ('grayscale_real', gray_manual),     # 2번째: Y채널 흑백
        ('realbgr', restored_manual),        # 3번째: 복원된 컬러
    ]
    gap = 20                     # 창 사이의 가로 여백(픽셀)
    width = img.shape[1]         # 이미지 가로 크기 (shape = (높이, 너비, 채널))
    for i, (title, image) in enumerate(windows):
        cv.namedWindow(title)                            # 창을 먼저 생성
        cv.moveWindow(title, i * (width + gap), 0)       # 이미지 너비+여백만큼 오른쪽으로 이동
        cv.imshow(title, image)                          # 해당 창에 영상 출력

    # 키 입력이 있을 때까지 창을 계속 띄워둠 (0은 무한 대기)
    cv.waitKey(0)
    # 열려 있는 모든 OpenCV 창을 닫음
    cv.destroyAllWindows()
