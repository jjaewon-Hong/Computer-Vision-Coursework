"""
Report #2 - 구현 3
time 모듈을 사용하여 OpenCV 함수(구현1)와 수식으로 직접 구현한 함수(구현2)의
실행시간을 비교하고, 두 결과(원영상 / Y영상 / 복원된 RGB영상)를 디스플레이
"""

import os                            # 파일 경로 처리용 표준 라이브러리
import time                          # 실행시간 측정을 위한 time.time() 함수 제공

import cv2 as cv                     # 이미지 입출력(imread)과 화면 출력(imshow)에 사용
import numpy as np                   # 이미지 바이트를 배열로 읽어 imdecode에 넘기기 위해 사용
import task2_1 as opencv_impl          # 구현 1: OpenCV 함수(cvtColor)를 이용한 변환 함수(rgb_to_gray_to_rgb_opencv)를 가져옴
import task2_2 as manual_impl          # 구현 2: 수식을 직접 구현한 변환 함수(rgb_to_gray_to_rgb_manual)를 가져옴


def main():
    # 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡음
    imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    # 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드함
    img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

    # ---------- 구현 1: OpenCV 함수를 이용한 변환 + 실행시간 측정 ----------
    start = time.time()  # 측정 시작 시각 기록
    gray_opencv, restored_opencv = opencv_impl.rgb_to_gray_to_rgb_opencv(img)
    opencv_time = time.time() - start  # 종료 시각 - 시작 시각 = 소요 시간(초)

    # ---------- 구현 2: 수식을 이용한 직접 구현 + 실행시간 측정 ----------
    start = time.time()  # 측정 시작 시각 기록
    gray_manual, restored_manual = manual_impl.rgb_to_gray_to_rgb_manual(img)
    manual_time = time.time() - start  # 종료 시각 - 시작 시각 = 소요 시간(초)

    # 두 방식의 실행시간을 소수점 4자리까지 출력하여 비교
    print(f'opencv_time : {opencv_time:.4f}')
    print(f'manual_time : {manual_time:.4f}')

    # ---------- 결과 비교 디스플레이 ----------
    # imshow만 호출하면 창 위치를 OS가 정하므로, namedWindow로 창을 먼저 만들고
    # moveWindow로 좌표를 직접 지정해서 원하는 순서대로 왼쪽부터 나란히 배치됨
    # 윗줄: 원본 + 구현1(OpenCV) 결과 / 아랫줄: 구현2(수식) 결과를 같은 열에 맞춰 배치하여
    # 흑백 영상끼리, 복원 영상끼리 위아래로 바로 비교할 수 있게 함
    height, width = img.shape[:2]   # 이미지 세로/가로 크기
    gap_x = 20                      # 창 사이의 가로 여백(픽셀)
    gap_y = 70                      # 창 사이의 세로 여백(창 제목 표시줄 높이를 감안)

    # (창 이름, 영상, 열 번호, 행 번호)
    windows = [
        ('original', img, 0, 0),                    # 1행 1열: 원본
        ('grayscale_opencv', gray_opencv, 1, 0),    # 1행 2열: 구현1 - Y채널 흑백
        ('repair', restored_opencv, 2, 0),          # 1행 3열: 구현1 - 복원된 컬러
        ('grayscale_real', gray_manual, 1, 1),      # 2행 2열: 구현2 - Y채널 흑백
        ('realbgr', restored_manual, 2, 1),         # 2행 3열: 구현2 - 복원된 컬러
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
