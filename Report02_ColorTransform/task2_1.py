"""
Report #2 - 구현 1
OpenCV 함수(cvtColor)를 이용하여 BGR -> YCrCb 변환 후 Y채널을 흑백 영상으로 추출하고,
YCrCb(Y, Cr, Cb)를 다시 BGR로 복원(RGB -> YCbCr -> RGB 왕복 변환)하는 프로그램
"""

import os           # 파일 경로 처리용 표준 라이브러리
import cv2 as cv     # OpenCV 라이브러리: 이미지 입출력, 색공간 변환(cvtColor), 화면 출력(imshow) 등에 사용
import numpy as np   # 이미지 바이트를 배열로 읽어 imdecode에 넘기기 위해 사용


def rgb_to_gray_to_rgb_opencv(bgr_img):
    """OpenCV cvtColor를 이용해 BGR -> YCrCb -> Y채널(흑백) 추출 -> BGR 복원"""

    # BGR -> YCrCb 변환 (OpenCV 내장 함수 이용)
    # OpenCV는 이미지를 기본적으로 B, G, R 순서로 저장하므로 COLOR_BGR2YCrCb를 사용
    ycrcb = cv.cvtColor(bgr_img, cv.COLOR_BGR2YCrCb)

    # 변환된 YCrCb 영상을 채널별로 분리 (순서: Y, Cr, Cb)
    # Cr, Cb 채널은 흑백 영상 추출에는 필요 없으므로 _cr, _cb로 받아 미사용 처리
    y, _cr, _cb = cv.split(ycrcb)

    # Y채널만 추출한 흑백 영상 (밝기 정보만 담고 있어 그대로 흑백 영상처럼 보임)
    gray = y.copy()

    # 복원: YCrCb -> BGR (Y, Cr, Cb 채널을 모두 이용한 완전한 역변환)
    # 분리하지 않고 원래의 ycrcb(Y, Cr, Cb 모두 포함)를 그대로 역변환에 사용
    restored_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

    return gray, restored_bgr


# 이 파일을 직접 실행했을 때만 아래 코드가 동작 (다른 파일에서 import 될 때는 실행되지 않음)
if __name__ == '__main__':
    # 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡음
    imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    # 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드함
    img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

    # 위에서 정의한 함수를 실행하여 흑백(Y채널) 영상과 복원된 컬러 영상을 얻음
    gray_opencv, restored_opencv = rgb_to_gray_to_rgb_opencv(img)

    # 원본, 흑백(Y채널), 복원된 영상을 각각 별도의 창으로 띄워서 비교
    # imshow만 호출하면 창 위치를 OS가 정하므로, namedWindow로 창을 먼저 만들고
    # moveWindow로 좌표를 직접 지정해서 원하는 순서대로 왼쪽부터 나란히 배치됨
    windows = [
        ('original', img),                      # 1번째: 원본
        ('grayscale_opencv', gray_opencv),      # 2번째: Y채널 흑백
        ('repair_opencv', restored_opencv),     # 3번째: 복원된 컬러
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
