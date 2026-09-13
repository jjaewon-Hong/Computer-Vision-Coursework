import os               # 파일 경로 처리용 표준 라이브러리
import numpy as np # 행렬 연산용 라이브러리 (OpenCV 이미지는 numpy 배열로 저장됨)
import cv2 as cv   # OpenCV 라이브러리

# 현재 작업 디렉터리와 무관하게, 이 스크립트가 있는 폴더 기준으로 경로를 잡는다
imgfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image.png')

# 이미지 컬러 (BGR 3채널)로 읽어 넘파이 배열 형태로 변환
# 경로에 한글이 있으면 cv.imread 가 실패하므로, 파일을 바이트로 읽어 디코드한다
img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), cv.IMREAD_COLOR)

cv.imshow('img', img) # 'img'라는 이름의 창에 이미지를 표시
cv.waitKey(0) # 키 입력이있을 때까지 대기 (무한 대기)
