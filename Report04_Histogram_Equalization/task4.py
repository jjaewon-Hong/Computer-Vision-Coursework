"""
Report #4 - Histogram Equalization ([구현 1] + [구현 2] + 실행시간 비교)

RGB(원본) -> YCbCr -> Y(밝기) 정보만 Histogram Equalization -> RGB(복원)
경로로 컬러 영상을 처리하고, 원본 RGB 와 복원된 RGB 사이의 PSNR 을 구한다.
(Cb, Cr 색차 정보는 건드리지 않으므로 색상은 유지되고 명암 대비만 개선됨)

[구현 1]: OpenCV 의 equalizeHist 함수를 이용하여 구현
[구현 2]: 강의자료의 수식을 이용하여 Histogram Equalization 을 직접 구현
[결과 비교]: 두 구현의 PSNR 이 동일한지 확인하고, time 모듈로 실행시간을 비교하며,
            원본 / [구현 1] 복원 영상 / [구현 2] 복원 영상을 디스플레이

[Histogram Equalization 수식]
h(i) = round( ( cdf(i) - cdf_min ) / ( N - cdf_min ) x L )
  cdf(i) : 밝기값 0 ~ i 까지의 누적 화소 수 (cumulative distribution function)
  cdf_min: cdf 값 중 0 이 아닌 최소값 (영상에 실제로 존재하는 가장 어두운 밝기의 cdf)
  N      : 영상의 전체 화소 수 (가로 x 세로)
  L      : 최대 밝기값 (8bit 영상이므로 255)

[PSNR]
PSNR = 10 * log10( MAX^2 / MSE ),  MAX = 255
"""

import os            # 파일 경로 처리용 표준 라이브러리
import time          # 실행시간 측정을 위한 time.time() 함수 제공

import cv2 as cv     # OpenCV: 이미지 입출력, 색공간 변환(cvtColor), equalizeHist, PSNR, 화면 출력
import numpy as np   # 히스토그램/누적합 계산, LUT 적용 등 배열 연산

# [구현 2] 수식을 이용한 Histogram Equalization 직접 구현
#          OpenCV 의 equalizeHist 를 쓰지 않고, 위 수식을 그대로 코드로 옮겨 구현

def equalize_hist_manual(y_channel, L=255):
    # 1) 히스토그램 계산: 밝기값 0~255 가 각각 몇 개의 화소인지 센다
    #    np.bincount 는 0 부터의 정수 개수를 한 번에 세어주며, minlength=256 으로 칸 수를 고정
    hist = np.bincount(y_channel.ravel(), minlength=256)

    # 2) 누적 히스토그램(cdf) 계산: hist 를 앞에서부터 차례로 더해 나간 값
    cdf = np.cumsum(hist)

    # 3) 수식에 필요한 값들
    N = y_channel.size            # 전체 화소 수 (가로 x 세로)
    cdf_min = cdf[cdf > 0][0]     # cdf 값 중 0 이 아닌 최소값

    # 4) 수식 적용: 밝기값 0~255 를 새로운 밝기값으로 바꿔주는 변환표(LUT, Look-Up Table) 생성
    #    화소마다 반복문을 도는 대신 256칸짜리 표를 한 번만 만들어 두고 재사용한다
    lut = np.round((cdf - cdf_min) / (N - cdf_min) * L)

    # 계산 결과가 0~255 를 벗어날 수 있으므로 클리핑 후 8bit 정수형으로 변환
    lut = np.clip(lut, 0, L).astype(np.uint8)

    # 5) 원래 밝기값을 인덱스로 사용해 LUT 에서 새 밝기값을 꺼내옴 (영상 전체를 한 번에 변환)
    return lut[y_channel]

# RGB(원본) -> YCbCr -> Y 정보만 Histogram Equalization -> RGB(복원)
# 색공간 변환은 두 구현이 동일하게 OpenCV 함수를 사용하고,
# equalize_func 로 넘겨준 Histogram Equalization 함수만 바뀌도록 해서 순수하게 그 부분만 비교

def equalize_on_y(bgr_img, equalize_func):
    # BGR -> YCrCb 변환 (OpenCV 는 이미지를 B, G, R 순서로 저장하므로 COLOR_BGR2YCrCb 사용)
    ycrcb = cv.cvtColor(bgr_img, cv.COLOR_BGR2YCrCb)

    # 채널 분리 (순서: Y, Cr, Cb) - Y(밝기)만 평활화하고 Cr, Cb(색차)는 그대로 둔다
    y, cr, cb = cv.split(ycrcb)

    # 넘겨받은 Histogram Equalization 함수만 실행시간 측정
    start = time.time()                  # 측정 시작 시각 기록
    y_eq = equalize_func(y)              # <- [구현 1] 또는 [구현 2] 함수
    elapsed = time.time() - start        # 종료 시각 - 시작 시각 = 소요 시간(초)

    # 평활화된 Y 와 원래의 Cr, Cb 를 다시 합쳐 YCrCb -> BGR 로 복원
    restored_bgr = cv.cvtColor(cv.merge([y_eq, cr, cb]), cv.COLOR_YCrCb2BGR)

    return y_eq, restored_bgr, elapsed

# [결과 비교] [구현 1]과 [구현 2]의 PSNR 값이 동일한지 확인하고,
#             time 모듈로 측정한 실행시간을 비교

def main():
    # 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡음
    imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    # 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드함
    img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

    # [구현 1] OpenCV equalizeHist 함수 이용
    y_opencv, restored_opencv, opencv_time = equalize_on_y(img, cv.equalizeHist)

    # [구현 2] 수식 직접 구현 함수 이용
    y_manual, restored_manual, manual_time = equalize_on_y(img, equalize_hist_manual)

    # 원본 RGB 와 복원된 RGB 사이의 PSNR (두 구현의 값이 동일한지 확인)
    psnr_opencv = cv.PSNR(img, restored_opencv, 255)
    psnr_manual = cv.PSNR(img, restored_manual, 255)

    # 두 방식의 PSNR 값과 실행시간을 출력하여 비교
    print(f'PSNR equalized by OpenCV function : {psnr_opencv:.4f}')
    print(f'PSNR equalized by formula         : {psnr_manual:.4f}')
    print(f'opencv_time = {opencv_time:.4f}')
    print(f'manual_time = {manual_time:.4f}')

    # 두 구현의 Y 결과가 화소 단위로 완전히 같은지 추가 확인
    # (같은 수식을 사용하므로 same result 는 True, max difference 는 0 이 나와야 함)
    print(f'same result : {np.array_equal(y_opencv, y_manual)}')
    print(f'max difference : {int(np.max(cv.absdiff(y_opencv, y_manual)))}')

    # ---------- 결과 비교 디스플레이 ----------
    # Y 영상이 아니라, Y 정보만 Histogram Equalization 한 뒤 RGB 로 복원한 컬러 영상을 출력
    # imshow만 호출하면 창 위치를 OS가 정하므로, namedWindow로 창을 먼저 만들고
    # moveWindow로 좌표를 직접 지정해서 원하는 순서대로 배치함
    height, width = img.shape[:2]   # 이미지 세로/가로 크기
    gap_x = 20                      # 창 사이의 가로 여백(픽셀)
    gap_y = 70                      # 창 사이의 세로 여백(창 제목 표시줄 높이를 감안)

    right_x = width + gap_x         # 오른쪽 열의 x 좌표
    row_h = height + gap_y          # 한 행이 차지하는 높이
    center_y = row_h // 2           # 원본을 오른쪽 두 창의 가운데 높이에 맞춤

    # (창 이름, 영상, x 좌표, y 좌표)
    # 왼쪽에 원본, 오른쪽에 두 구현 결과를 위아래로 쌓아 한눈에 비교
    windows = [
        ('original', img, 0, center_y),                           # 왼쪽 가운데: 원본
        ('equalized_opencv', restored_opencv, right_x, 0),        # 오른쪽 위: [구현 1]
        ('equalized_manual', restored_manual, right_x, row_h),    # 오른쪽 아래: [구현 2]
    ]
    for title, image, x, y in windows:
        cv.namedWindow(title)         # 창을 먼저 생성
        cv.moveWindow(title, x, y)    # 지정한 좌표로 이동
        cv.imshow(title, image)       # 해당 창에 영상 출력

    # 키 입력이 있을 때까지 모든 창을 계속 띄워둠 (0은 무한 대기)
    cv.waitKey(0)
    # 열려 있는 모든 OpenCV 창을 닫음
    cv.destroyAllWindows()

# 이 파일을 직접 실행했을 때만 main() 함수가 동작 (다른 파일에서 import 될 때는 실행되지 않음)
if __name__ == '__main__':
    main()
