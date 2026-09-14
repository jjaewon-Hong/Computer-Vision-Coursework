"""
Report #2 - BGR <-> YCbCr 색공간 변환 ([구현 1] + [구현 2] + 실행시간 비교)

[구현 1]: OpenCV 함수(cvtColor)를 이용하여 BGR -> YCrCb 변환 후 Y채널을 흑백 영상으로
          추출하고, 다시 BGR로 복원
[구현 2]: 강의자료의 수식을 이용하여 BGR -> YCbCr, YCbCr -> BGR 변환을 직접 구현
[구현 3]: time 모듈로 두 방식의 실행시간을 측정하여 비교하고,
          원영상 / Y영상 / 복원된 RGB영상을 차례로 디스플레이

[BGR -> YCbCr]  (강의자료 For SDTV 수식, full range 0~255)
Y  =  0.299R + 0.587G + 0.114B
Cb = -0.172R - 0.339G + 0.511B + 128
Cr =  0.511R - 0.428G - 0.083B + 128

[YCbCr -> BGR]
R = Y + 1.371(Cr - 128)
G = Y - 0.698(Cr - 128) - 0.336(Cb - 128)
B = Y + 1.732(Cb - 128)
"""

import os            # 파일 경로 처리용 표준 라이브러리
import time          # 실행시간 측정을 위한 time.time() 함수 제공

import cv2 as cv     # OpenCV: 이미지 입출력, 색공간 변환(cvtColor), 화면 출력(imshow)
import numpy as np   # 채널 분리, 행렬 연산(수식 계산), clip, dstack 등 배열 연산

# [구현 1] OpenCV 함수(cvtColor)를 이용한 구현 - 시작
#          RGB -> YCbCr 변환 -> Y채널만 추출해 흑백 영상 -> 다시 RGB로 복원

def rgb_to_gray_to_rgb_opencv(bgr_img):
    # BGR -> YCrCb 변환 (OpenCV 내장 함수 이용)
    # OpenCV는 이미지를 기본적으로 B, G, R 순서로 저장하므로 COLOR_BGR2YCrCb를 사용
    ycrcb = cv.cvtColor(bgr_img, cv.COLOR_BGR2YCrCb)

    # 변환된 YCrCb 영상을 채널별로 분리 (순서: Y, Cr, Cb)
    # Cr, Cb 채널은 흑백 영상 추출에는 필요 없으므로 _cr, _cb로 받아 미사용 처리
    y, _cr, _cb = cv.split(ycrcb)

    # Y채널만 추출한 흑백 영상 (밝기 정보만 담고 있어 그대로 흑백 영상처럼 보임)
    gray = y.copy()

    # 복원: YCrCb -> BGR (Y, Cr, Cb 채널을 모두 이용한 완전한 역변환)
    restored_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

    return gray, restored_bgr

# [구현 2] 수식을 이용한 직접 구현 - 시작
#          OpenCV의 cvtColor를 쓰지 않고, 강의자료 수식으로 직접 계산
#          (bgr2ycrcb_manual / ycrcb2bgr_manual / rgb_to_gray_to_rgb_manual)

def bgr2ycrcb_manual(bgr_img):
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

    # 위에서 정의한 함수로 BGR -> YCbCr 변환 수행
    Y, Cb, Cr = bgr2ycrcb_manual(bgr_img)

    # Y채널만 추출한 흑백 영상 (디스플레이용, 8bit 정수형으로 변환)
    gray = Y.astype(np.uint8)

    # 복원: YCbCr -> BGR (Y, Cb, Cr 채널을 모두 이용한 완전한 역변환)
    restored_bgr = ycrcb2bgr_manual(Y, Cb, Cr)

    return gray, restored_bgr

# [구현 3] time 모듈을 이용한 실행시간 비교 - 시작
#          위의 [구현 1]과 [구현 2] 함수를 각각 호출해 실행시간을 측정/비교하고
#          원영상 / Y영상 / 복원된 RGB영상을 차례로 디스플레이

def main():
    # 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡음
    imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    # 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드함
    img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

    # [구현 1] OpenCV 함수 호출 + 실행시간 측정
    start = time.time()  # 측정 시작 시각 기록
    gray_opencv, restored_opencv = rgb_to_gray_to_rgb_opencv(img)   # <- [구현 1] 함수
    opencv_time = time.time() - start  # 종료 시각 - 시작 시각 = 소요 시간(초)

    # [구현 2] 수식 직접 구현 함수 호출 + 실행시간 측정
    start = time.time()  # 측정 시작 시각 기록
    gray_manual, restored_manual = rgb_to_gray_to_rgb_manual(img)   # <- [구현 2] 함수
    manual_time = time.time() - start  # 종료 시각 - 시작 시각 = 소요 시간(초)

    # 두 방식의 실행시간을 소수점 4자리까지 출력하여 비교
    print(f'opencv_time : {opencv_time:.4f}')
    print(f'manual_time : {manual_time:.4f}')

    # ---------- 결과 비교 디스플레이 ----------
    # imshow만 호출하면 창 위치를 OS가 정하므로, namedWindow로 창을 먼저 만들고
    # moveWindow로 좌표를 직접 지정해서 원하는 순서대로 배치함
    # 윗줄: 원본 + [구현 1](OpenCV) 결과 / 아랫줄: [구현 2](수식) 결과를 같은 열에 맞춰 배치하여
    # 흑백 영상끼리, 복원 영상끼리 위아래로 바로 비교할 수 있게 함
    height, width = img.shape[:2]   # 이미지 세로/가로 크기
    gap_x = 20                      # 창 사이의 가로 여백(픽셀)
    gap_y = 70                      # 창 사이의 세로 여백(창 제목 표시줄 높이를 감안)

    # (창 이름, 영상, 열 번호, 행 번호)
    windows = [
        ('original', img, 0, 0),                    # 1행 1열: 원본
        ('grayscale_opencv', gray_opencv, 1, 0),    # 1행 2열: [구현 1] - Y채널 흑백
        ('repair', restored_opencv, 2, 0),          # 1행 3열: [구현 1] - 복원된 컬러
        ('grayscale_real', gray_manual, 1, 1),      # 2행 2열: [구현 2] - Y채널 흑백
        ('realbgr', restored_manual, 2, 1),         # 2행 3열: [구현 2] - 복원된 컬러
    ]
    for title, image, col, row in windows:
        cv.namedWindow(title)                                                    # 창을 먼저 생성
        cv.moveWindow(title, col * (width + gap_x), row * (height + gap_y))      # 격자 위치로 이동
        cv.imshow(title, image)                                                  # 해당 창에 영상 출력

    # 키 입력이 있을 때까지 모든 창을 계속 띄워둠 (0은 무한 대기)
    cv.waitKey(0)
    # 열려 있는 모든 OpenCV 창을 닫음
    cv.destroyAllWindows()

# 이 파일을 직접 실행했을 때만 main() 함수가 동작 (다른 파일에서 import 될 때는 실행되지 않음)
if __name__ == '__main__':
    main()
