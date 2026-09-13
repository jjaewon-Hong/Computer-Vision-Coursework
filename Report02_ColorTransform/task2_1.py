"""
Report #2 - 구현 1
OpenCV 함수(cvtColor)를 이용하여 BGR -> YCrCb 변환 후 Y채널을 흑백 영상으로 추출하고,
YCrCb(Y, Cr, Cb)를 다시 BGR로 복원(RGB -> YCbCr -> RGB 왕복 변환)하는 프로그램
"""

import cv2 as cv  # OpenCV 라이브러리: 이미지 입출력, 색공간 변환(cvtColor), 화면 출력(imshow) 등에 사용


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
    imgfile = 'image.png'  # 불러올 이미지 파일 경로

    # 이미지를 컬러(BGR, 3채널)로 읽어오기
    img = cv.imread(imgfile, cv.IMREAD_COLOR)

    # 이미지 로드 실패 시(경로 오류 등) 예외를 발생시켜 원인을 바로 알 수 있도록 처리
    if img is None:
        raise FileNotFoundError(f'이미지를 불러올 수 없습니다: {imgfile}')

    # 위에서 정의한 함수를 실행하여 흑백(Y채널) 영상과 복원된 컬러 영상을 얻음
    gray_opencv, restored_opencv = rgb_to_gray_to_rgb_opencv(img)

    # 원본, 흑백(Y채널), 복원된 영상을 각각 별도의 창으로 띄워서 비교
    cv.imshow('original', img)
    cv.imshow('grayscale_opencv', gray_opencv)
    cv.imshow('repair_opencv', restored_opencv)

    # 키 입력이 있을 때까지 창을 계속 띄워둠 (0은 무한 대기)
    cv.waitKey(0)
    # 열려 있는 모든 OpenCV 창을 닫음
    cv.destroyAllWindows()
