"""
Report #2 - 구현 3
time 모듈을 사용하여 OpenCV 함수(구현1)와 수식으로 직접 구현한 함수(구현2)의
실행시간을 비교하고, 두 결과(원영상 / Y영상 / 복원된 RGB영상)를 디스플레이
"""

import time                          # 실행시간 측정을 위한 time.time() 함수 제공

import cv2 as cv                     # 이미지 입출력(imread)과 화면 출력(imshow)에 사용
import task2_1 as opencv_impl          # 구현 1: OpenCV 함수(cvtColor)를 이용한 변환 함수(rgb_to_gray_to_rgb_opencv)를 가져옴
import task2_2 as manual_impl          # 구현 2: 수식을 직접 구현한 변환 함수(rgb_to_gray_to_rgb_manual)를 가져옴


def main():
    imgfile = 'image.png'  # 불러올 이미지 파일 경로

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    img = cv.imread(imgfile, cv.IMREAD_COLOR)

    # 이미지 로드 실패 시(경로 오류 등) 예외를 발생시켜 원인을 바로 알 수 있도록 처리
    if img is None:
        raise FileNotFoundError(f'이미지를 불러올 수 없습니다: {imgfile}')

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
    # 원본 영상
    cv.imshow('original', img)

    # 구현 1(OpenCV) 결과: Y채널 흑백 영상, 복원된 컬러 영상
    cv.imshow('grayscale_opencv', gray_opencv)
    cv.imshow('repair', restored_opencv)

    # 구현 2(수식 직접 구현) 결과: Y채널 흑백 영상, 복원된 컬러 영상
    cv.imshow('grayscale_real', gray_manual)
    cv.imshow('realbgr', restored_manual)

    # 키 입력이 있을 때까지 모든 창을 계속 띄워둠 (0은 무한 대기)
    cv.waitKey(0)
    # 열려 있는 모든 OpenCV 창을 닫음
    cv.destroyAllWindows()


# 이 파일을 직접 실행했을 때만 main() 함수가 동작 (다른 파일에서 import 될 때는 실행되지 않음)
if __name__ == '__main__':
    main()
