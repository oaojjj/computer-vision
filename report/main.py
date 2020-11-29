import numpy as np
from PIL import Image
import cv2
import random
import cmath

CONST_LOW = 0
CONST_HIGH = 255

embo_mask_hpf = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
embo_mask_hpf2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

mask_lpf_3x3 = np.array(
    [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])

mask_lpf_5x5 = np.array(
    [[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
     [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
     [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]])

gaussian_mask_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

gaussian_mask_5x5 = np.array(
    [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
     [1, 4, 7, 4, 1]])

gaussian_mask_7x7 = np.array([[0, 0, 1, 2, 1, 0, 0],
                              [0, 3, 13, 22, 13, 3, 0],
                              [1, 13, 59, 97, 59, 13, 1],
                              [2, 22, 97, 159, 97, 22, 2],
                              [1, 13, 59, 97, 59, 13, 1],
                              [0, 3, 13, 22, 13, 3, 0],
                              [0, 0, 1, 2, 1, 0, 0]])

gaussian_mask_5x5_2 = np.array(
    [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0],
     [0, 0, -1, 0, 0]])

gaussian_mask_5x5_3 = np.array(
    [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, 24, -1, -1], [-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1]])

sharpening_mask_lpf_3x3 = np.array(
    [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sharpening_mask_hpf2_3x3 = np.array(
    [[-2 / 9, -2 / 9, -2 / 9], [-2 / 9, 16 / 9, -2 / 9], [-2 / 9, -2 / 9, -2 / 9]])

laplacian_mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
laplacian_mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
laplacian_mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
laplacian_mask5x5 = np.array(
    [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

difference_of_gaussian_mask_9x9 = np.array(
    [[0, 0, 0, -1, -1, -1, 0, 0, 0],
     [0, -2, -3, -3, -3, -3, -3, -2, 0],
     [0, -3, -2, -1, -1, -1, -2, -3, 0],
     [-1, -3, -1, 9, 9, 9, -1, -3, -1],
     [-1, -3, -1, 9, 19, 9, -1, -3, -1],
     [-1, -3, -1, 9, 9, 9, -1, -3, -1],
     [0, -3, -2, -1, -1, -1, -2, -3, 0],
     [0, -2, -3, -3, -3, -3, -3, -2, 0],
     [0, 0, 0, -1, -1, -1, 0, 0, 0]])

difference_of_gaussian_mask_7x7 = np.array([[0, 0, -1, -1, -1, 0, 0],
                                            [0, -2, -3, -3, -3, -2, 0],
                                            [-1, -3, 5, 5, 5, -3, -1],
                                            [-1, -3, 5, 16, 5, -3, -1],
                                            [-1, -3, 5, 5, 5, -3, -1],
                                            [0, -2, -3, -3, -3, -2, 0],
                                            [0, 0, -1, -1, -1, 0, 0]])


# (1) 포인트 프로그램
# 1. 이진영상 프로그램 만들기
def threshold(image, number):
    # 결과이미지 저장할 변수 생성
    image_output = np.zeros(image.shape, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 픽셀의 값이 number 보다 작으면 0 대입
            if image[i, j] < number:
                image_output[i, j] = CONST_LOW
            # 픽셀의 값이 number 보다 크다면 255 대입
            else:
                image_output[i, j] = CONST_HIGH

    return np.uint8(image_output)


def my_threshold(image, number):
    image_new = threshold(image, number)
    cv2.imshow("threshold", np.hstack([image, image_new]))
    cv2.waitKey(0)


# 2. 감마보정 변환 곡선 프로그램 𝛾 = 0.25, 𝛾 = 2.5
def my_imadjust(image, gamma=1):
    # 감마 0.1, 0.7, 1.5, 2.5에 따른 결과값 생성
    for gamma in [0.1, 0.7, 1.5, 2.5]:
        invGamma = 1.0 / gamma  # 감마값 생성
        image_output = np.array(image, copy=True)
        cv2.putText(image_output, "gamma={}".format(gamma), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # 감마보정 공식을 통해 픽셀값을 결정
                image_output[i, j] = \
                    ((image_output[i, j] / CONST_HIGH) ** invGamma) * CONST_HIGH

        cv2.imshow("imadjust", np.hstack([image, np.uint8(image_output)]))
        cv2.waitKey(0)


# 3. 반전 영상 프로그램 g(x,y)=255-f(x,y)
def my_reverse(image):
    # 결과이미지 저장할 변수 생성
    image_new = np.array(image, copy=True)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 255 - 현재의 픽셀 값
            image_new[i, j] = CONST_HIGH - image_new[i, j]

    cv2.imshow("reverse", np.hstack([image, image_new]))
    cv2.waitKey(0)


# 4. 강조 영상 프로그램
def my_highlight(image, start, end):
    # 결과이미지 저장할 변수 생성
    image_new = np.array(image, copy=True)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # start 이상 end 이하의 값들은 255로 변환
            if start <= image_new[i, j] <= end:
                image_new[i, j] = CONST_HIGH

    cv2.imshow("highlight", np.hstack([image, image_new]))
    cv2.waitKey(0)


# (2) 영상화질개선 프로그램
# 1. 기본명암대비 스트레칭 기법: 공식 이용
def my_contrast_stretching(image):
    # 최대, 최소값 초기화
    high = 0
    low = 255

    # 결과이미지 저장할 변수 생성
    image_new = np.zeros(image.shape, dtype=np.float64)

    # 이미지에서 최대값과 최소값을 찾기
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > high:
                high = image[i, j]
            if image[i, j] < low:
                low = image[i, j]

    # 스트레칭 기법 공식을 적용하여 새로운 픽셀값을 대입
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_new[i, j] = (image[i, j] - low) / (high - low) * CONST_HIGH

    image_new = np.uint8(image_new)
    cv2.imshow("contrast_stretching", np.hstack([image, image_new]))
    cv2.waitKey(0)


# 2. 앤드-인-탐색 기법: 공식 이용
def my_and_in(image):
    # 최대, 최소값 초기화
    high = 0
    low = 255

    # 결과이미지 저장할 변수 생성
    image_new = np.zeros(image.shape, dtype=np.float64)

    # 이미지에서 최대값과 최소값을 찾기
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > high:
                high = image[i, j]
            if image[i, j] < low:
                low = image[i, j]

    # 최소값 이하는 0 최대값 이상은 255 그사이의 값은 공식을 적용하여 대입
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            old_pixel = image[i, j]
            if old_pixel <= low:
                image_new[i, j] = 0
            elif low <= old_pixel <= high:
                image_new[i, j] = (old_pixel - low) / (high - low) * CONST_HIGH
            elif old_pixel >= high:
                image_new[i, j] = CONST_HIGH

    image_new = np.uint8(image_new)
    cv2.imshow("and_in", np.hstack([image, image_new]))
    cv2.waitKey(0)


# 3. 히스토그램 평활화 기법
def my_histogram_equalization(image):
    image_origin = np.array(image, copy=True)

    # 빈도수 조사
    histogram = [0] * 256
    create_histogram(histogram, image)

    # 누적 히스토그램 생성
    sum_of_histogram = [0] * 256
    create_sum_of_histogram(histogram, sum_of_histogram)

    # 입력 영상을 평활화된 영상으로 출력
    equalization(image, image.shape[0] * image.shape[1], sum_of_histogram)

    cv2.imshow("histogram_equalization", np.hstack([image_origin, image]))
    cv2.waitKey(0)


# 히스토그램(빈도수 조사)
def create_histogram(histogram, image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            value = image[i, j]
            histogram[value] += 1


# 히스토그램 누적합
def create_sum_of_histogram(histogram, sum_of_histogram):
    sum = 0.0
    for i in range(0, 256):
        sum += histogram[i]
        sum_of_histogram[i] = sum


# 평활화
def equalization(image, size, sum_of_histogram):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = image[i, j]
            # normalization
            image[i, j] = sum_of_histogram[temp] * 255.0 / size


# 4. 히스토그램 명세화 기법
def my_histogram_specification(image, image2):
    top = CONST_HIGH
    bottom = top - 1

    image_origin = np.array(image, copy=True)

    # 빈도수 조사
    histogram = [0] * 256
    want_histogram = [0] * 256
    create_histogram(histogram, image)
    create_histogram(want_histogram, image2)

    # 누적 히스토그램 조사
    sum_of_histogram = [0] * 256
    sum_of_want_histogram = [0] * 256
    create_sum_of_histogram(histogram, sum_of_histogram)
    create_sum_of_histogram(want_histogram, sum_of_want_histogram)

    # 원본 영상의 평활화
    equalization(image, image.shape[0] * image.shape[1], sum_of_histogram)

    d_min = sum_of_want_histogram[0]
    d_max = sum_of_want_histogram[255]

    # 원하는 영상을 평활화
    sum_of_new_histogram = [0] * 256
    for i in range(0, 256):
        sum_of_new_histogram[i] = ((sum_of_want_histogram[i] - d_min) * CONST_HIGH / (d_max - d_min))

    # 룩업테이블 만들기 (역평활화)
    table = [0] * 256
    while True:
        for i in range(round(sum_of_new_histogram[bottom]), round(sum_of_new_histogram[top]) + 1):
            table[i] = top
        top = bottom
        bottom -= 1
        if bottom < -1:
            break;

    image_output = np.array(image, copy=True)

    # mapping output_image (룩업테이블을 보고 역변환, 명세화)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = image_output[i, j]
            image_output[i, j] = table[temp]

    cv2.imshow("histogram_specification", np.hstack([image_origin, image2, image_output]))
    cv2.waitKey(0)


# (3) 영상화질개선 프로그램
# 1. 영상의 산술연산(덧셈연산)
def my_add_image(image, image2):
    # 결과 저장용 변수
    image_new = np.array(image2, copy=True)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 두 영상의 픽셀값을 더함
            value = int(image[i, j]) + int(image2[i, j])
            # 정규화 -> 대입
            if value > CONST_HIGH:
                image_new[i, j] = CONST_HIGH
            else:
                image_new[i, j] = value

    cv2.imshow("add_image", np.hstack([image, image2, image_new]))
    cv2.waitKey(0)


# 영상의 산술연산(뺄셈연산)
def my_subtract_image(image, image2):
    # 결과 저장용 변수
    image_new = np.array(image, copy=True)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 두 영상의 픽셀값을 빼줌
            value = int(image[i, j]) - int(image2[i, j])
            # 정규화 -> 대입
            if value < CONST_LOW:
                image_new[i, j] = CONST_LOW
            else:
                image_new[i, j] -= image2[i, j]

    cv2.imshow("subtract_image", np.hstack([image, image2, image_new]))
    cv2.waitKey(0)


# 2. gaussian_noise 평균으로 잡음제거 영상처리
def gaussian_noise(image, n):
    # 노이즈 저장 변수
    img_noise = []
    for i in range(0, n):
        # 가우시안 노이즈 이미지 생성
        img_noise.append(create_gaussian_noise(image, 32))
        cv2.waitKey(0)

    # 가우시안 노이즈 이미지들의 필터링 결과 이미지(필터링)
    filter_image = img_avg(image, img_noise, n)
    print("가우시안 노이즈" + str(n) + "개 이미지 생성")
    return np.uint8(filter_image)


# 가우시안 노이즈 생성
def create_gaussian_noise(image, std):
    # 노이즈가 발생한 이미지를 저장하는 변수
    image_noisy = np.zeros(image.shape, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 난수 생성 (float64)
            normal = np.random.normal()
            # 표준편차 32
            noise = std * normal
            image_noisy[i, j] = image[i, j] + noise
            if image_noisy[i, j] > CONST_HIGH:
                image_noisy[i, j] = CONST_HIGH
            elif image_noisy[i, j] < CONST_LOW:
                image_noisy[i, j] = CONST_LOW
    return np.uint8(image_noisy)


# 가우시안 노이즈 이미지들의 평균 연산(필터링)
def img_avg(origin_img, image, n):
    img = np.zeros(origin_img.shape, dtype=np.float64)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            value = 0
            # 노이즈 픽셀값을 전부 더하고 평균 연산
            for img_noisy in image:
                value += img_noisy[i, j]
            img[i, j] = value / n

    return img


def my_gaussian_noise():
    img = cv2.imread('color_lenna.jpg', cv2.IMREAD_GRAYSCALE)
    img_gn_1 = gaussian_noise(img, 1)
    cv2.putText(img_gn_1, "N=1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    img_gn_2 = gaussian_noise(img, 4)
    cv2.putText(img_gn_2, "N=4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    img_gn_3 = gaussian_noise(img, 8)
    cv2.putText(img_gn_3, "N=8", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.imshow('gaussian_noise', np.hstack([img_gn_1, img_gn_2, img_gn_3]))
    cv2.waitKey(0)


# 3. 영상의 논리연산
# - AND 연산, OR 연산, X-OR 연산
def my_and(image, image2):
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    image2 = cv2.resize(image2, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    image = threshold(image, 128)
    image2 = threshold(image2, 128)

    img_and = np.zeros(image.shape, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_and[i, j] = image[i, j] & image2[i, j]

    cv2.imshow('and', np.hstack([image, image2, img_and]))
    cv2.waitKey(0)


def my_or(image, image2):
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    image2 = cv2.resize(image2, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    image = threshold(image, 128)
    image2 = threshold(image2, 128)

    img_or = np.zeros(image.shape, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_or[i, j] = image[i, j] | image2[i, j]

    cv2.imshow('or', np.hstack([image, image2, img_or]))
    cv2.waitKey(0)


def my_xor(image, image2):
    image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    image2 = cv2.resize(image2, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    image = threshold(image, 128)
    image2 = threshold(image2, 128)

    img_xor = np.zeros(image.shape, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_xor[i, j] = image[i, j] ^ image2[i, j]

    cv2.imshow('xor', np.hstack([image, image2, img_xor]))
    cv2.waitKey(0)


# 4. 비트 플레인
def bit_plane(image, n):
    bit = 0b01 << n  # n칸 쉬프트

    img_new = np.zeros(image.shape, dtype=np.float64)

    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            pixel = image[i, j]
            img_new[i, j] = int(pixel) & bit

    return img_new


def my_bit_plane():
    img = cv2.imread('color_lenna.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    image_bp_list = []
    cv2.imshow('origin', img)

    for n in range(0, 8):
        image_bp_list.append(bit_plane(img, n))
        cv2.imshow(str(n + 1) + 'bit image', image_bp_list[n])
        cv2.waitKey(1)

    cv2.waitKey(0)


# (4) 영역처리(공간필터링)프로그램
# 1. 엠보싱 효과 프로그램
def my_embossing(image):
    image_origin = np.array(image, copy=True)
    image_out_put = mask_process(image, embo_mask_hpf, 1)
    image_out_put2 = mask_process(image, embo_mask_hpf2, 1)

    cv2.imshow('embossing', np.hstack([image_origin, np.uint8(image_out_put), np.uint8(image_out_put2)]))
    cv2.waitKey(0)


# 엠보싱 회선 처리 함수
def mask_process(image, embo_mask, padding):
    image_out_put = np.zeros(image.shape, dtype=np.float64)
    image_padding = np.zeros([image.shape[0] + padding * 2, image.shape[1] + padding * 2], dtype=np.float64)

    # zero_padding 을 제외한 값 복사
    image_padding[padding: padding + image.shape[0], padding: padding + image.shape[1]] = image.copy()

    # 회선연산
    for i in range(image_out_put.shape[0]):
        for j in range(image_out_put.shape[1]):
            sum = 0.0
            for n in range(len(embo_mask)):
                for m in range(len(embo_mask)):
                    sum += embo_mask[n][m] * image_padding[i + n][j + m]
            temp = sum + 128
            if temp > CONST_HIGH:
                image_out_put[i][j] = CONST_HIGH
            elif temp < CONST_LOW:
                image_out_put[i][j] = CONST_LOW
            else:
                image_out_put[i][j] = temp

    return np.uint8(image_out_put)


# 2. 블러링 프로그램
def my_blurring(image):
    image_origin = image.copy()
    image_out_put = mask_process2(image, mask_lpf_3x3, 1)

    image = image_origin.copy()
    image_out_put2 = mask_process2(image, mask_lpf_5x5, 2)

    cv2.putText(image_out_put, "3x3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(image_out_put2, "5x5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    cv2.imshow('embossing', np.hstack([image_origin, image_out_put, image_out_put2]))
    cv2.waitKey(0)


# 회선 처리 함수
def mask_process2(image, embo_mask, padding):
    image_out_put = np.zeros(image.shape, dtype=np.float64)
    image_padding = np.zeros([image.shape[0] + padding * 2, image.shape[1] + padding * 2], dtype=np.float64)

    # zero_padding 을 제외한 값 복사
    image_padding[padding: padding + image.shape[0], padding: padding + image.shape[1]] = image.copy()

    # 회선연산
    for i in range(image_out_put.shape[0]):
        for j in range(image_out_put.shape[1]):
            sum = 0.0
            for n in range(len(embo_mask)):
                for m in range(len(embo_mask)):
                    sum += embo_mask[n][m] * image_padding[i + n][j + m]
            # 회선 처리 결과가 0~255 사이 값이 되도록 한다.
            if sum > CONST_HIGH:
                image_out_put[i][j] = CONST_HIGH
            elif sum < CONST_LOW:
                image_out_put[i][j] = CONST_LOW
            else:
                image_out_put[i][j] = sum

    return np.uint8(image_out_put)


# 3. 가우시안 스무딩 필터링 프로그램
def my_gaussian_smoothing(image):
    image_origin = image.copy()
    image_out_put = gaussian_mask_process(image, gaussian_mask_5x5, 2, 273)

    cv2.putText(image_out_put, "gaussian_smoothing_5x5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    cv2.imshow('gaussian_smoothing', np.hstack([image_origin, image_out_put]))
    cv2.waitKey(0)


# 가우시안 회선 처리 함수
def gaussian_mask_process(image, embo_mask, padding, g_sum):
    image_out_put = np.zeros(image.shape, dtype=np.float64)
    image_padding = np.zeros([image.shape[0] + padding * 2, image.shape[1] + padding * 2], dtype=np.float64)

    # zero_padding 을 제외한 값 복사
    image_padding[padding: padding + image.shape[0], padding: padding + image.shape[1]] = image.copy()

    # 회선연산
    for i in range(image_out_put.shape[0]):
        for j in range(image_out_put.shape[1]):
            sum = 0.0
            for n in range(len(embo_mask)):
                for m in range(len(embo_mask)):
                    sum += embo_mask[n][m] * image_padding[i + n][j + m]
            # 회선 처리 결과를 마스크의 합으로 나누고 결과가 0~255 사이 값이 되도록 한다.
            sum /= g_sum
            if sum > CONST_HIGH:
                image_out_put[i][j] = CONST_HIGH
            elif sum < CONST_LOW:
                image_out_put[i][j] = CONST_LOW
            else:
                image_out_put[i][j] = sum

    return np.uint8(image_out_put)


# 4. 샤프닝 프로그램
def my_sharpening(image):
    image_origin = image.copy()
    image_out_put = mask_process2(image, sharpening_mask_lpf_3x3, 1)

    cv2.putText(image_out_put, "3x3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    cv2.imshow('sharpening', np.hstack([image_origin, image_out_put]))
    cv2.waitKey(0)


# 5. 고주파 통과필터를 이용한 샤프닝 처리 프로그램
def my_high_pass_filter_sharpening(image):
    image_origin = image.copy()
    image_out_put = mask_process2(image, sharpening_mask_hpf2_3x3, 1)

    cv2.putText(image_out_put, "high_pass_filter_3x3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    cv2.imshow('high_filter_sharpening', np.hstack([image_origin, image_out_put]))
    cv2.waitKey(0)


# 6. 저주파 통과필터를 이용한 샤프닝 처리 프로그램
def my_low_pass_filter_sharpening(image):
    image_origin = image.copy()

    image_unsharp_masking = mask_process2(image, mask_lpf_3x3, 1)
    image_high_boost = image_unsharp_masking.copy()

    # unsharp masking (원 영상) - (저주파 통과 필터링 결과 영상)
    for i in range(image_origin.shape[0]):
        for j in range(image_origin.shape[1]):
            temp = round(image_origin[i][j]) - round(image_unsharp_masking[i][j])
            if temp > CONST_HIGH:
                image_unsharp_masking[i][j] = CONST_HIGH
            elif temp < CONST_LOW:
                image_unsharp_masking[i][j] = CONST_LOW
            else:
                image_unsharp_masking[i][j] = temp + 10

    α = 2.1
    # high boost α(원 영상) - (저주파 통과 필터링 결과 영상)
    for i in range(image_high_boost.shape[0]):
        for j in range(image_high_boost.shape[1]):
            temp = α * round(image_origin[i][j]) - round(image_high_boost[i][j])
            if temp > CONST_HIGH:
                image_high_boost[i][j] = CONST_HIGH
            elif temp < CONST_LOW:
                image_high_boost[i][j] = CONST_LOW
            else:
                image_high_boost[i][j] = temp

    cv2.putText(image_unsharp_masking, "unsharp masking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_high_boost, "high boost (a=" + str(α) + ")", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 3)

    cv2.imshow('low_filter_sharpening',
               np.hstack([image_origin, np.uint8(image_unsharp_masking), np.uint8(image_high_boost)]))
    cv2.waitKey(0)


# 7. 가우시안 잡음을 생성하여 LPF를 사용하여 잡음 제거
def my_gaussian_low_filter(image):
    image_gaussian = create_gaussian_noise(image, 32)
    image_output = mask_process2(image_gaussian, mask_lpf_5x5, 4)
    cv2.imshow('gaussian_low_filter',
               np.hstack([image_gaussian, image_output]))
    cv2.waitKey(0)


# (5) 칼라변환 프로그램
# 1. RGB  CMY 로 칼라 변환
def my_rgb_change_cmy(image):
    imageRGB = [image.copy(), image.copy(), image.copy()]
    imageCMY = []

    # rgb추출
    extraction_rgb(imageRGB)

    imageCMY.append(imageRGB[2] + imageRGB[1])
    imageCMY.append(imageRGB[0] + imageRGB[2])
    imageCMY.append(imageRGB[0] + imageRGB[1])

    cv2.putText(image, "origin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("rgb_change_cmy_origin", image)

    cv2.putText(imageRGB[0], "red", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(imageRGB[1], "green", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(imageRGB[2], "blue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("rgb", np.hstack([imageRGB[0], imageRGB[1], imageRGB[2]]))

    cv2.putText(imageCMY[0], "cyan", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(imageCMY[1], "magenta", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(imageCMY[2], "yellow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("cmy", np.hstack([imageCMY[0], imageCMY[1], imageCMY[2]]))
    cv2.waitKey(0)


def extraction_rgb(image):
    # Red만 추출
    image[0][:, :, 1], image[0][:, :, 0] = 0, 0

    # Green만 추출
    image[1][:, :, 0], image[1][:, :, 2] = 0, 0

    # Blue만 추출
    image[2][:, :, 1], image[2][:, :, 2] = 0, 0


def check_value(value):
    if value > CONST_HIGH:
        return CONST_HIGH
    elif value < CONST_LOW:
        return CONST_LOW
    else:
        return value


# 2. RGB  HIS 로 칼라 변환
def my_rgb_change_HIS(image):
    imageHIS = np.zeros(image.shape, dtype=np.float32)
    image_copy = image.copy() / 255.0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            imageHIS[i, j] = rgb_change_his(image_copy[i, j])

    cv2.putText(image, "origin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("rgb_change_his_origin", image)

    img_h = np.uint8(np.clip(imageHIS[:, :, 0] * 255.0, 0, 255))
    img_i = np.uint8(np.clip(imageHIS[:, :, 1] * 255.0, 0, 255))
    img_s = np.uint8(np.clip(imageHIS[:, :, 2] * 255.0, 0, 255))

    cv2.putText(img_h, "hue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img_i, "intensity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img_s, "saturation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("his", np.hstack([img_h, img_i, img_s]))
    cv2.waitKey(0)


def rgb_change_his(rgb):
    b, g, r = rgb[0], rgb[1], rgb[2]

    # 명도(intensity)
    I = np.mean(rgb)

    # 채도(saturation)
    S = 1 - (min(rgb) / I)

    # 색상(hue)
    temp = np.divide(((r - g) + (r - b)), (2 * np.sqrt((r - g) * (r - g) + (r - b) * (g - b))))
    H = np.arccos(temp) * 180 / np.pi

    if b > g:
        H = 360 - H
    H /= 360

    return np.array([check_value(H), check_value(I), check_value(S)], dtype=np.float32)


# 3. RGB  YCbCr 로 칼라 변환
def my_rgb_change_YCrCb(image):
    imageYCrCb = np.zeros(image.shape, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            imageYCrCb[i, j] = rgb_change_YCrCb(image[i, j])

    img_y = np.uint8(np.clip(imageYCrCb[:, :, 0], 0, 255))
    img_cr = np.uint8(np.clip(imageYCrCb[:, :, 1], 0, 255))
    img_cb = np.uint8(np.clip(imageYCrCb[:, :, 2], 0, 255))

    cv2.putText(image, "origin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("my_rgb_change_YCbCr", image)

    # 명도
    cv2.putText(img_y, "Y", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # 붉은색 정보
    cv2.putText(img_cr, "Cr", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # 푸른색 정보
    cv2.putText(img_cb, "Cb", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("YCbCr", np.hstack([img_y, img_cr, img_cb]))
    cv2.waitKey(0)


def rgb_change_YCrCb(rgb):
    b, g, r = rgb[0], rgb[1], rgb[2]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cr = 0.500 * r - 0.419 * g - 0.0813 * b + 180
    Cb = -0.169 * r - 0.331 * g + 0.500 * b + 180

    return np.array([Y, Cr, Cb], dtype=np.float32)


# (6) 에지연산자 프로그램 레포트
# 1. 유사연산자 기법 프로그램
def my_homogeneity_operator(image):
    # zero_padding 을 제외한 값 복사
    image_temp = np.zeros([image.shape[0] + 2, image.shape[1] + 2], dtype=np.float64)
    image_output = np.zeros([image.shape[0], image.shape[1]], dtype=np.float64)
    image_temp[1: 1 + image.shape[0], 1: 1 + image.shape[1]] = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            max = 0.0  # 최대 값 초기화
            for n in range(3):
                for m in range(3):
                    value = abs(image_temp[i + 1, j + 1] - image_temp[i + n, j + n])
                    if value >= max:
                        max = value
            image_output[i, j] = max

    image_output = np.clip(image_output, 0, 255)
    cv2.imshow("homogeneity_operator", np.hstack([image, np.uint8(image_output)]))
    cv2.waitKey(0)


# 2. 차연산자 기법 프로그램
def my_difference_operator(image):
    # zero_padding 을 제외한 값 복사
    image_temp = np.zeros([image.shape[0] + 2, image.shape[1] + 2], dtype=np.float64)
    image_output = np.zeros([image.shape[0], image.shape[1]], dtype=np.float64)
    image_temp[1: 1 + image.shape[0], 1: 1 + image.shape[1]] = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            max = 0.0  # 최대 값 초기화
            for n in range(2):
                for m in range(2):
                    k = n * 3 + m
                    if k < 4:
                        # 최대값을 찾음
                        if abs(image_temp[i + 2 - n, j + 2 - m] - image_temp[i + n, j + n]) >= max:
                            max = abs(image_temp[i + 2 - n, j + 2 - m] - image_temp[i + n, j + n])

            image_output[i, j] = max

    image_output = np.clip(image_output, 0, 255)
    cv2.imshow("difference_operator", np.hstack([image, np.uint8(image_output)]))
    cv2.waitKey(0)


# 3. 로버츠, 프리윗, 소벨 연산자 프로그램
def my_roberts_operator(image):
    roberts_row_mask = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    roberts_col_mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

    image_row = mask_process2(image, roberts_row_mask, 1)
    image_col = mask_process2(image, roberts_col_mask, 1)
    image_sum = image_row + image_col
    cv2.putText(image_row, "row", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_col, "col", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_sum, "sum", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("roberts_operator", np.hstack([image, np.uint8(image_row), ]))
    cv2.imshow("roberts_operator2", np.hstack([np.uint8(image_col), np.uint8(image_sum)]))
    cv2.waitKey(0)


def my_prewitt_operator(image):
    prewitt_row_mask = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_col_mask = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    image_row = mask_process2(image, prewitt_row_mask, 1)
    image_col = mask_process2(image, prewitt_col_mask, 1)
    image_sum = image_row + image_col
    cv2.putText(image_row, "row", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_col, "col", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_sum, "sum", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("prewitt_operator", np.hstack([image, np.uint8(image_row), ]))
    cv2.imshow("prewitt_operator2", np.hstack([np.uint8(image_col), np.uint8(image_sum)]))
    cv2.waitKey(0)


def my_sobel_operator(image):
    sobel_row_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_col_mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    image_row = mask_process2(image, sobel_row_mask, 1)
    image_col = mask_process2(image, sobel_col_mask, 1)
    image_sum = image_row + image_col
    cv2.putText(image_row, "row", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_col, "col", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_sum, "sum", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("sobel_operator", np.hstack([image, np.uint8(image_row), ]))
    cv2.imshow("sobel_operator2", np.hstack([np.uint8(image_col), np.uint8(image_sum)]))
    cv2.waitKey(0)


# 4. 라플라시안 연산자 프로그램
def my_laplacian_operator(image):
    image_origin = image.copy()
    image_out_put = mask_process2(image, laplacian_mask1, 1)
    image_out_put2 = mask_process2(image, laplacian_mask2, 1)
    image_out_put3 = mask_process2(image, laplacian_mask3, 1)
    cv2.putText(image_out_put, "mask1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_out_put2, "mask2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_out_put3, "mask3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow('laplacian_operator', np.hstack([image_origin, image_out_put]))
    cv2.imshow('laplacian_operator2', np.hstack([image_out_put2, image_out_put3]))
    cv2.waitKey(0)


# 5. LoG 연산자 프로그램
def my_laplacian_of_gaussian_operator(image):
    image_origin = image.copy()
    image_out_put = mask_process2(image, laplacian_mask5x5, 2)

    cv2.imshow('laplacian_of_gaussian_operator', np.hstack([image_origin, image_out_put]))
    cv2.waitKey(0)


# 6. Dog 연산자 프로그램
def my_difference_of_gaussian_operator(image):
    image_origin = image.copy()

    image_gaussian_low = gaussian_mask_process(image, gaussian_mask_3x3, 1, 16)
    image_gaussian_high = gaussian_mask_process(image, gaussian_mask_5x5, 2, 273)

    # low - high
    image_output = image_gaussian_low - image_gaussian_high

    image_output = np.clip(image_output, 0, 255)
    cv2.imshow('difference_of_gaussian_operator', np.hstack([image_origin, np.uint8(image_output)]))
    cv2.waitKey(0)


# (7) 에지연산자 프로그램 레포트
# 1. Kuwahara 필터링 프로그램
def my_kuwahara_filter(image, pad):
    image_output = np.zeros(image.shape, dtype=np.float64)  # 결과를 저장할 변수 생성

    # zero_padding 이미지 생성
    image_zero_padding = np.zeros([image.shape[0] + pad * 2, image.shape[1] + pad * 2], dtype=np.float64)
    image_zero_padding[pad: pad + image.shape[0], pad: pad + image.shape[1]] = image

    # 분산 구하는법 -> 편차제곱의 평균
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            shape = (i + pad, j + pad)

            # 3x3 마스크로 지역을 나눠서 저장
            mask_region = np.zeros((4, pad + 1, pad + 1))
            mask_region[0] = image_zero_padding[i:(shape[0] + 1), j:(shape[1] + 1)]
            mask_region[1] = image_zero_padding[i:(shape[0] + 1), shape[1]:(shape[1] + pad + 1)]
            mask_region[2] = image_zero_padding[shape[0]:(shape[0] + 1 + pad), j:(shape[1] + 1)]
            mask_region[3] = image_zero_padding[shape[0]:(shape[0] + 1 + pad), shape[1]:(shape[1] + 1 + pad)]

            # 최소 분산 값을 가진 마스크 영역을 찾기
            # 각각 분산값 저장
            var = [np.var(mask_region[0]),
                   np.var(mask_region[1]),
                   np.var(mask_region[2]),
                   np.var(mask_region[3])]

            # 분산 최소값의 인덱스 위치
            min_index = np.argwhere(var == np.min(var))[0, 0]

            # 최소 분산값의 마스크의 평균을 화소에 삽입
            image_output[i, j] = np.sum(mask_region[min_index]) / 9

    image_output = np.uint8(np.clip(image_output, 0, 255))
    cv2.imshow('kuwahara_filter', np.hstack([image, image_output]))
    cv2.waitKey(0)


# 2. Nagao-Matsuyama 필터링 프로그램
def my_nagao_matsuyama(image, pad):
    image_output = np.zeros(image.shape, dtype=np.float64)  # 결과를 저장할 변수 생성

    # zero_padding 이미지 생성
    image_zero_padding = np.zeros([image.shape[0] + pad * 2, image.shape[1] + pad * 2], dtype=np.float64)
    image_zero_padding[pad: pad + image.shape[0], pad: pad + image.shape[1]] = image

    # 분산 구하는법 -> 편차제곱의 평균
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            shape = (i + pad, j + pad)

            mask_region = np.zeros((8, 7))
            # 3x3 마스크로 동서남북 지역을 나눠서 저장
            mask_region[0] = np.delete(
                np.ravel(image_zero_padding[i:(shape[0] + 1), j:(shape[1] + 1)]), (6, 8))
            mask_region[1] = np.delete(
                np.ravel(image_zero_padding[i:(shape[0] + 1), shape[1]:(shape[1] + pad + 1)]), (0, 6))
            mask_region[2] = np.delete(
                np.ravel(image_zero_padding[shape[0]:(shape[0] + 1 + pad), j:(shape[1] + 1)]), (0, 2))
            mask_region[3] = np.delete(
                np.ravel(image_zero_padding[shape[0]:(shape[0] + 1 + pad), shape[1]:(shape[1] + 1 + pad)]), (2, 8))

            # 대각선 영역 저장
            # 우선 3x3 넘파이로 저장을 하고 1차원으로 만들어주고 특정한 인덱스에 위치한 값을 delete 해서 대각선 배열을 만든다.
            mask_region[4] = np.delete(
                np.ravel(image_zero_padding[i:(shape[0] + 1), j:(shape[1] + 1)]), (2, 6))
            mask_region[5] = np.delete(
                np.ravel(image_zero_padding[i:(shape[0] + 1), shape[1]:(shape[1] + pad + 1)]), (0, 8))
            mask_region[6] = np.delete(
                np.ravel(image_zero_padding[shape[0]:(shape[0] + 1 + pad), j:(shape[1] + 1)]), (2, 6))
            mask_region[7] = np.delete(
                np.ravel(image_zero_padding[shape[0]:(shape[0] + 1 + pad), shape[1]:(shape[1] + 1 + pad)]), (0, 8))

            # 최소 분산 값을 가진 마스크 영역을 찾기
            var = [np.var(mask_region[0]),
                   np.var(mask_region[1]),
                   np.var(mask_region[2]),
                   np.var(mask_region[3]),
                   np.var(mask_region[4]),
                   np.var(mask_region[5]),
                   np.var(mask_region[6]),
                   np.var(mask_region[7])]

            # 분산 최소값의 인덱스 위치
            min_index = np.argwhere(var == np.min(var))[0, 0]

            # 최소 분산값의 마스크의 평균을 화소에 삽입
            image_output[i, j] = np.sum(mask_region[min_index]) / 7

    image_output = np.uint8(np.clip(image_output, 0, 255))
    cv2.imshow('nagao_matsuyama', np.hstack([image, image_output]))
    cv2.waitKey(0)


# 3. 미디언 필터 프로그램
def my_median_filter(image):
    image_output = np.zeros(image.shape, dtype=np.float64)  # 결과를 저장할 변수 생성
    image_noise = create_salt_and_paper(image, 10000)  # salt&paper 노이즈 이미지 생성
    image_noise_zero_padding = np.zeros([image.shape[0] + 2, image.shape[1] + 2],
                                        dtype=np.float64)  # zero_padding 이미지 생성
    image_noise_zero_padding[1:-1, 1:-1] = image_noise  # zero_padding을 제외한 픽셀 복사

    filter_list = [0] * 9  # 중앙값을 찾기 위해 리스트 선언
    for i in range(1, image_noise_zero_padding.shape[0] - 1):
        for j in range(1, image_noise_zero_padding.shape[1] - 1):
            filter_list[0] = image_noise_zero_padding[i - 1, j - 1]
            filter_list[1] = image_noise_zero_padding[i - 1, j]
            filter_list[2] = image_noise_zero_padding[i - 1, j + 1]
            filter_list[3] = image_noise_zero_padding[i, j - 1]
            filter_list[4] = image_noise_zero_padding[i, j]
            filter_list[5] = image_noise_zero_padding[i, j + 1]
            filter_list[6] = image_noise_zero_padding[i + 1, j - 1]
            filter_list[7] = image_noise_zero_padding[i + 1, j]
            filter_list[8] = image_noise_zero_padding[i + 1, j + 1]
            filter_list.sort()
            image_output[i - 1, j - 1] = filter_list[4]  # 정렬한 값의 중앙값을 대입입

    image_output = np.uint8(np.clip(image_output, 0, 255))
    cv2.imshow('median_filter', np.hstack([image_noise, image_output]))
    cv2.waitKey(0)


# 4. 하이브리드 미디언 필터 프로그램
def my_hybrid_median_filter(image):
    image_output = np.zeros(image.shape, dtype=np.float64)  # 결과를 저장할 변수 생성
    image_noise = create_salt_and_paper(image, 10000)  # salt&paper 노이즈 이미지 생성
    image_noise_zero_padding = \
        np.zeros([image.shape[0] + 2, image.shape[1] + 2], dtype=np.float64)  # zero_padding 이미지 생성
    image_noise_zero_padding[1:-1, 1:-1] = image_noise  # zero_padding을 제외한 픽셀 복사

    filter_list2, filter_list1 = [0] * 5, [0] * 5  # 중앙값을 찾기 위해 리스트 선언
    for i in range(1, image_noise_zero_padding.shape[0] - 1):
        for j in range(1, image_noise_zero_padding.shape[1] - 1):
            # list1은 대각선 list2는 역대각선 픽셀을 계산한다
            filter_list1[0] = image_noise_zero_padding[i - 1, j - 1]
            filter_list1[1] = image_noise_zero_padding[i - 1, j + 1]
            filter_list1[2] = image_noise_zero_padding[i, j]
            filter_list1[3] = image_noise_zero_padding[i + 1, j - 1]
            filter_list1[4] = image_noise_zero_padding[i + 1, j + 1]

            filter_list2[0] = image_noise_zero_padding[i - 1, j]
            filter_list2[1] = image_noise_zero_padding[i, j - 1]
            filter_list2[2] = image_noise_zero_padding[i, j]
            filter_list2[4] = image_noise_zero_padding[i, j + 1]
            filter_list2[3] = image_noise_zero_padding[i + 1, j]

            filter_list1.sort()
            filter_list2.sort()
            mid = image_noise_zero_padding[i, j]
            image_output[i - 1, j - 1] = np.median([filter_list1[2], filter_list2[2], mid])  # 정렬한 값의 중앙값을 대입입

    image_output = np.uint8(np.clip(image_output, 0, 255))
    cv2.imshow('hybrid_median_filter', np.hstack([image_noise, image_output]))
    cv2.waitKey(0)


def create_salt_and_paper(image, n):
    for k in range(n):
        i = random.randrange(0, image.shape[0])
        j = random.randrange(0, image.shape[1])

        noise = random.randrange(0, 2)

        value = 0 if noise == 0 else 255
        image[i, j] = value
    return image


# 5. 최대, 최소값 필터링 프로그램
def my_max_min_filter(image):
    image_output_max = np.zeros(image.shape, dtype=np.float64)  # max_filter 결과를 저장할 변수 생성
    image_output_min = np.zeros(image.shape, dtype=np.float64)  # min_filter 결과를 저장할 변수 생성
    image_noise_zero_padding = np.zeros([image.shape[0] + 2, image.shape[1] + 2],
                                        dtype=np.float64)  # zero_padding 이미지 생성
    image_noise_zero_padding[1:-1, 1:-1] = image  # zero_padding을 제외한 픽셀 복사

    filter_list = [0] * 9  # 중앙값을 찾기 위해 리스트 선언
    for i in range(1, image_noise_zero_padding.shape[0] - 1):
        for j in range(1, image_noise_zero_padding.shape[1] - 1):
            filter_list[0] = image_noise_zero_padding[i - 1, j - 1]
            filter_list[1] = image_noise_zero_padding[i - 1, j]
            filter_list[2] = image_noise_zero_padding[i - 1, j + 1]
            filter_list[3] = image_noise_zero_padding[i, j - 1]
            filter_list[4] = image_noise_zero_padding[i, j]
            filter_list[5] = image_noise_zero_padding[i, j + 1]
            filter_list[6] = image_noise_zero_padding[i + 1, j - 1]
            filter_list[7] = image_noise_zero_padding[i + 1, j]
            filter_list[8] = image_noise_zero_padding[i + 1, j + 1]
            filter_list.sort()
            image_output_max[i - 1, j - 1] = np.max(filter_list)  # 최대값 출력
            image_output_min[i - 1, j - 1] = np.min(filter_list)  # 최소값 출력

    image_output_max = np.uint8(np.clip(image_output_max, 0, 255))
    image_output_min = np.uint8(np.clip(image_output_min, 0, 255))
    cv2.putText(image_output_max, "max", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(image_output_min, "min", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow('max_min_filter', np.hstack([image, image_output_max, image_output_min]))
    cv2.waitKey(0)


# 6. alpha-trimmed mean 프로그램
def my_alpha_trimmed_mean_filter(image):
    image_noise = create_salt_and_paper(image, 50000)  # salt&paper 노이즈 이미지 생성
    cv2.imshow('alpha_trimmed_noise', image_noise)
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        image_output = alpha_trimmed_mean_filter(image, alpha, image_noise.copy())
        cv2.imshow('alpha_trimmed' + str(alpha), image_output)
        cv2.waitKey(0)


def alpha_trimmed_mean_filter(image, alpha, image_noise):
    image_output = np.zeros(image.shape, dtype=np.float64)  # 결과를 저장할 변수 생성
    image_noise_zero_padding = np.zeros([image.shape[0] + 2, image.shape[1] + 2],
                                        dtype=np.float64)  # zero_padding 이미지 생성
    image_noise_zero_padding[1:-1, 1:-1] = image_noise  # zero_padding을 제외한 픽셀 복사

    filter_list = [0] * 9  # 중앙값을 찾기 위해 리스트 선언
    for i in range(1, image_noise_zero_padding.shape[0] - 1):
        for j in range(1, image_noise_zero_padding.shape[1] - 1):
            filter_list[0] = image_noise_zero_padding[i - 1, j - 1]
            filter_list[1] = image_noise_zero_padding[i - 1, j]
            filter_list[2] = image_noise_zero_padding[i - 1, j + 1]
            filter_list[3] = image_noise_zero_padding[i, j - 1]
            filter_list[4] = image_noise_zero_padding[i, j]
            filter_list[5] = image_noise_zero_padding[i, j + 1]
            filter_list[6] = image_noise_zero_padding[i + 1, j - 1]
            filter_list[7] = image_noise_zero_padding[i + 1, j]
            filter_list[8] = image_noise_zero_padding[i + 1, j + 1]
            filter_list.sort()
            # 양쪽으로 잘라야 하는 범위 구하기
            value = int(alpha * 9)
            # alpha 값에 따른 픽셀값 구하기
            if 0 <= alpha <= 0.1:
                image_output[i - 1, j - 1] = np.average(filter_list)
            elif 0.1 < alpha <= 0.5:
                image_output[i - 1, j - 1] = np.average(filter_list[value:-value])
            else:
                image_output[i - 1, j - 1] = filter_list[4]

    image_output = np.uint8(np.clip(image_output, 0, 255))

    cv2.putText(image_output, "alpha" + str(alpha), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return image_output


# (8) 에지연산자 프로그램 코딩 레포트
# 1. 2D DFT를 이용한 2차원 주파수 변환 및 역변환

# 기하학적 변환 프로그램 코딩 레포트
# 1-1. 보간법 프로그램(최근접 이웃보간법)

def my_interpolate(image, new_size):
    # 기본 shape(size) 저장
    old_shape = image.shape

    # 비율값
    row_ratio, col_ratio = np.array(new_size) / np.array(old_shape)

    # 행 단위 보간(비율단위로 원하는 사이즈 픽셀값 생성)
    row_index = (np.ceil(range(1, 1 + int(old_shape[0] * row_ratio)) / row_ratio) - 1).astype(int)

    # 열 단위 보간(비율단위로 원하는 사이즈 픽셀값 생성)
    col_index = (np.ceil(range(1, 1 + int(old_shape[1] * col_ratio)) / col_ratio) - 1).astype(int)

    # 비율에 맞게 행과 열에 각각 보간(인접한 값)하여 얻은값을 대입한다.
    image_output = image[:, row_index][col_index, :]

    cv2.imshow('interpolate_origin', image)
    cv2.imshow('interpolate_output', image_output)
    cv2.waitKey(0)


# 1-2. 보간법 프로그램(양선형 보간법)
# https://engineer-mole.tistory.com/13?category=911427
def my_bilinear_interpolate(image, scale):
    H, W = image.shape

    aH = int(scale * H)
    aW = int(scale * W)

    # 리사이싱된 이미지의 픽셀값 정의
    y = np.arange(aH).repeat(aW).reshape(aH, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    # 리사이징된 이미지 픽셀을 원본이미지의 픽셀 비율로 나누기
    y = (y / scale)
    x = (x / scale)

    # 픽셀값 반올림
    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    # 최소값 찾기
    ix = np.minimum(ix, W - 2)
    iy = np.minimum(iy, H - 2)

    # 거리 구하기
    dx = x - ix
    dy = y - iy

    dx = np.repeat(dx, 1, axis=-1)
    dy = np.repeat(dy, 1, axis=-1)

    # 보간값 구하기
    image_output = (1 - dx) * (1 - dy) * image[iy, ix] + \
                   dx * (1 - dy) * image[iy, ix + 1] + \
                   (1 - dx) * dy * image[iy + 1, ix] + \
                   dx * dy * image[iy + 1, ix + 1]

    image_output = np.clip(image_output, 0, 255)
    image_output = image_output.astype(np.uint8)

    cv2.imshow('bilinear_interpolate_origin', image)
    cv2.imshow('bilinear_interpolate_output', image_output)
    cv2.waitKey(0)


# 2. 회전, 스케일링, 이동 프로그램
# 2-1. 회전
def my_rotation(img, degree):
    height = img.shape[0]
    width = img.shape[1]
    image_output = np.zeros(img.shape, dtype=np.float)

    rad = float(degree * np.pi / 180.0)  # degree -> radian
    center_y, center_x = height / 2, width / 2  # 중심좌표

    for y in range(height):
        for x in range(width):
            # 회전된 좌표값 계산
            py = int((y - center_y) * np.cos(rad) - (x - center_x) * np.sin(rad) + center_y)
            px = int((y - center_y) * np.sin(rad) + (x - center_x) * np.sin(rad) + center_x)

            if (py < 0 or py >= height) or (px < 0 or px >= width):
                val = 0
            else:
                val = img[py, px]

            image_output[y, x] = val
            image_output = np.uint8(image_output)

    cv2.imshow('rotation_origin', img)
    cv2.imshow('rotation_output', image_output)
    cv2.waitKey(0)


# 2-2. 스케일링
def my_scaling(img, ratio):
    # 이미지 리사이징
    image_output = np.zeros((img.shape[0] * ratio, img.shape[1] * ratio), dtype=np.float)

    # 리사이징된 이미지의 shape를 행과 열로 나누어 반복
    for x in range(int(img.shape[0] * ratio)):
        for y in range(int(img.shape[1] * ratio)):
            # 픽셀값을 비율만큼 나누어 저장
            px = int(x / ratio)
            py = int(y / ratio)
            image_output[y, x] = img[py, px]

    cv2.imshow('scaling_origin', img)
    cv2.imshow('scaling_output', np.uint8(image_output))
    cv2.waitKey(0)


# 2-3. 이동
def my_translation(img, h_pos, w_pos):
    height = img.shape[0]
    width = img.shape[1]
    image_output = np.zeros(img.shape, dtype=np.float)

    for x in range(width - w_pos):
        for y in range(height - h_pos):
            image_output[x + w_pos][y + h_pos] = img[x, y]

    cv2.imshow('translation_origin', img)
    cv2.imshow('translation_output', np.uint8(image_output))
    cv2.waitKey(0)


# 3. 아핀 변환 프로그램
def my_affine(img, a, b, c, d, tx, ty):
    height, width = img.shape

    # temporary image
    temp_img = np.zeros((height + 2, width + 2), dtype=np.float32)
    temp_img[1:height + 1, 1:width + 1] = img

    # get new image shape
    height_new = np.round(height * d).astype(np.int)
    width_new = np.round(width * a).astype(np.int)
    image_output = np.zeros((height_new + 1, width_new + 1), dtype=np.float32)

    # get position of new image
    x_new = np.tile(np.arange(width_new), (height_new, 1))
    y_new = np.arange(height_new).repeat(width_new).reshape(height_new, -1)

    # get position of original image by affine
    adbc = a * d - b * c
    x = np.round((d * x_new - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1

    x = np.minimum(np.maximum(x, 0), width + 1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), height + 1).astype(np.int)

    # assgin pixcel to new image
    image_output[y_new, x_new] = temp_img[y, x]

    image_output = image_output[:height_new, :width_new]
    image_output = image_output.astype(np.uint8)

    cv2.imshow('affine_origin', img)
    cv2.imshow('affine_output', np.uint8(image_output))
    cv2.waitKey(0)


if __name__ == '__main__':
    img_lena = cv2.imread('color_lenna.jpg', cv2.IMREAD_GRAYSCALE)
    img_color_lena = cv2.imread('color_lenna.jpg')
    img_color_flower = cv2.imread('flower.jpg')
    img_gray_scale = cv2.imread('gray_scale_image.jpg', cv2.IMREAD_GRAYSCALE)
    img_gray_scale2 = cv2.imread('gray_scale_image2.jpg', cv2.IMREAD_GRAYSCALE)
    img_gray_scale3 = cv2.imread('gray_scale_image3.jpg', cv2.IMREAD_GRAYSCALE)
    img_black = cv2.imread('black_image.jpg', cv2.IMREAD_GRAYSCALE)
    img_stretching = cv2.imread('stretching_image.jpg', cv2.IMREAD_GRAYSCALE)
    img_semiconduct512 = cv2.imread('semiconduct512.jpg', cv2.IMREAD_GRAYSCALE)
    img_circle = cv2.imread('gray_scale_image_circle.jpg', cv2.IMREAD_GRAYSCALE)
    img_circle2 = cv2.imread('gray_scale_image_circle2.jpg', cv2.IMREAD_GRAYSCALE)
    img_embossing = cv2.imread('embossing_image.jpg', cv2.IMREAD_GRAYSCALE)

    img_lena = cv2.resize(img_lena, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_color_lena = cv2.resize(img_color_lena, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_color_flower = cv2.resize(img_color_flower, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_gray_scale = cv2.resize(img_gray_scale, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_gray_scale2 = cv2.resize(img_gray_scale2, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_gray_scale3 = cv2.resize(img_gray_scale3, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_black = cv2.resize(img_black, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_stretching = cv2.resize(img_stretching, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_semiconduct512 = cv2.resize(img_semiconduct512, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_circle = cv2.resize(img_circle, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_circle2 = cv2.resize(img_circle2, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    img_embossing = cv2.resize(img_embossing, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    # 1-1. 포인트 프로그램
    # my_threshold(img_lena, 130)
    # my_imadjust(img_lena)
    # my_reverse(img_lena)
    # my_highlight(img_lena, 125, 200)

    # 1-2. 영상화질개선 프로그램
    # my_contrast_stretching(img_lena)
    # my_and_in(img_lena)
    # my_histogram_equalization(img_lena)
    # my_histogram_specification(img_lena, img_gray_scale3)
    # my_histogram_specification(img_lena, img_black)

    # 1-3. 영상화질개선 프로그램
    # my_add_image(img_lena, img_circle)
    # my_subtract_image(img_lena, img_circle2)
    # my_subtract_image(img_circle, img_lena)
    # my_gaussian_noise()
    # my_and(img_lena, img_circle)
    # my_or(img_lena, img_circle)
    # my_xor(img_lena, img_circle)
    # my_bit_plane()

    # 2-1. 영역처리(공간필터링)프로그램
    # my_embossing(img_lena)
    # my_blurring(img_lena)
    # my_gaussian_smoothing(img_lena)
    # my_sharpening(img_lena)
    # my_high_pass_filter_sharpening(img_lena)
    # my_low_pass_filter_sharpening(img_lena)
    # my_gaussian_low_filter(img_lena)

    # 2-2. 칼라변환 프로그램
    # my_rgb_change_cmy(img_color_lena)
    # my_rgb_change_HIS(img_color_lena)
    # my_rgb_change_YCrCb(img_color_lena)

    # 3-1. 에지연산자 프로그램 레포트
    # my_homogeneity_operator(img_lena)
    # my_difference_operator(img_lena)
    # my_roberts_operator(img_lena)
    # my_prewitt_operator(img_lena)
    # my_sobel_operator(img_lena)
    # my_laplacian_operator(img_lena)
    # my_laplacian_of_gaussian_operator(img_lena)
    # my_difference_of_gaussian_operator(img_lena)

    # 3-2
    # my_kuwahara_filter(img_lena, 2)
    # my_nagao_matsuyama(img_lena, 2)
    # my_median_filter(img_lena)
    # my_hybrid_median_filter(img_lena)
    # my_max_min_filter(img_lena)
    # my_alpha_trimmed_mean_filter(img_lena)

    # 3-2

    # 기하학적 변환 프로그램 코딩 레포트
    # 4-1
    # my_interpolate(img_lena, 1024)
    # my_bilinear_interpolate(img_lena, 4)
    # my_rotation(img_lena, 30)
    # my_scaling(img_lena, 2)
    # my_translation(img_lena, 30, 130)
    my_affine(img_lena, a=1, b=0, c=0, d=1, tx=30, ty=-30)
