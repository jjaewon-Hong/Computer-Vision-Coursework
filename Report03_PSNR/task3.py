"""
Report03 - PSNR 구하기 ([구현 1] + [구현 2] + 실행시간 비교)

Report02 에서 만든 RGB(원본) -> YCbCr -> RGB(복원) 경로를 그대로 사용하여,
원본 영상과 복원 영상 사이의 PSNR을 두 가지 방법으로 구하고 결과를 비교한다.

[구현 1]: OpenCV 의 PSNR 함수(cv.PSNR)를 이용하여 구현
[구현 2]: 강의자료의 수식을 이용하여 PSNR 함수를 직접 구현
[구현 3]: time 모듈로 두 함수의 실행시간을 측정하여 비교

[MSE]  컬러 영상(3채널)의 경우
MSE = (1 / (3 x M x N)) * SUM_c SUM_j SUM_i ( I(i,j,c) - R(i,j,c) )^2

[PSNR]
PSNR = 10 * log10( MAX^2 / MSE ) = 20 * log10( MAX / sqrt(MSE) )
여기서 MAX 는 화소의 최대값인 255 이다.
"""

import os            # 파일 경로 처리용 표준 라이브러리
import time          # 실행시간 측정을 위한 time.time() 함수 제공

import cv2 as cv     # OpenCV: 이미지 입출력, 색공간 변환(cvtColor), PSNR, 화면 출력(imshow)
import numpy as np   # 채널 분리, 행렬 연산(수식 계산), clip, dstack 등 배열 연산

# [Report02 동일] RGB -> YCbCr -> RGB 복원 (PSNR 비교 대상 영상을 만드는 부분)
#             OpenCV 함수(cvtColor)를 이용한 변환/복원

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

# [Report02 동일] 수식을 이용한 직접 변환/복원
#                 OpenCV의 cvtColor를 쓰지 않고, 강의자료 수식으로 직접 계산
#                 (bgr2ycrcb_manual / ycrcb2bgr_manual / rgb_to_gray_to_rgb_manual)

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

# [구현 1] OpenCV 의 PSNR 함수를 이용한 구현 - 시작

def getPSNR_opencv(img1, img2):
    # cv.PSNR(src1, src2, R) : 두 영상의 PSNR(dB)을 계산해 주는 OpenCV 내장 함수
    # R 은 화소의 최대값(8bit 영상이므로 255), 생략해도 기본값이 255 이다.
    return cv.PSNR(img1, img2, 255)

# [구현 2] 수식을 이용한 직접 구현 - 시작
#          강의자료의 MSE / PSNR 수식을 그대로 코드로 옮겨 구현

def getPSNR_manual(img1, img2, max_val=255.0):
    # uint8 상태로 빼면 음수가 표현되지 않아 오차가 왜곡되므로 실수형(float64)으로 변환
    I = img1.astype(np.float64)   # 원본 영상
    R = img2.astype(np.float64)   # 복원 영상

    # 화소별 차이 -> 제곱 -> 전체 평균
    # np.mean 은 (3 x M x N) 개의 모든 원소로 나누므로 컬러 MSE 수식과 동일하다.
    diff = I - R
    mse = np.mean(diff * diff)

    # 두 영상이 완전히 같으면 MSE = 0 이 되어 log 계산이 불가능하므로 무한대로 처리
    if mse == 0:
        return float('inf')

    # PSNR = 10 * log10( MAX^2 / MSE )
    return 10.0 * np.log10((max_val ** 2) / mse)

# [구현 3] time 모듈을 이용한 실행시간 비교 - 시작
#          [구현 1]과 [구현 2]의 PSNR 값이 동일한지, 실행시간은 어떤지 비교

def main():
    # 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡음
    imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    # 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드함
    img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

    # Report02 의 수식 구현으로 RGB(원본) -> YCbCr -> RGB(복원) 을 수행하여 복원 영상을 얻음
    # (OpenCV 복원 영상으로 비교하려면 아래 한 줄을 rgb_to_gray_to_rgb_opencv(img) 로 바꾸면 됨)
    _gray, restored = rgb_to_gray_to_rgb_manual(img)

    # [구현 2] 직접 구현한 함수 호출 + 실행시간 측정
    start = time.time()                              # 측정 시작 시각 기록
    psnr_manual = getPSNR_manual(img, restored)      # <- [구현 2] 함수
    manual_time = time.time() - start                # 종료 시각 - 시작 시각 = 소요 시간(초)

    # [구현 1] OpenCV 함수 호출 + 실행시간 측정
    start = time.time()                              # 측정 시작 시각 기록
    psnr_opencv = getPSNR_opencv(img, restored)      # <- [구현 1] 함수
    opencv_time = time.time() - start                # 종료 시각 - 시작 시각 = 소요 시간(초)

    # 두 방식의 PSNR 값과 실행시간을 출력하여 비교
    print(f'PSNR implemented by function : {psnr_opencv:.4f}')
    print(f'manual_time : {manual_time:.4f}')
    print(f'PSNR implemented by formula  : {psnr_manual:.4f}')
    print(f'opencv_time : {opencv_time:.4f}')

# 이 파일을 직접 실행했을 때만 main() 함수가 동작 (다른 파일에서 import 될 때는 실행되지 않음)
if __name__ == '__main__':
    main()
