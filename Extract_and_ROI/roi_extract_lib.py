import cv2 as cv
import numpy as np

def find_contours(binary):
    # 查找图像轮廓及最大外轮廓
    # 输入二值图像binary
    # 返回所有轮廓及最大外轮廓索引contours,maxAreaIdx
    _, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # 查找最大轮廓
    maxAreaIdx = 0
    maxArea = 0.0
    for t in range(len(contours)):
        tempArea = abs(cv.contourArea(contours[t]))
        if tempArea > maxArea:
            maxArea = tempArea
            maxAreaIdx = t
    return contours, maxAreaIdx

def get_fourier_descriptor(border,proportion):
    # 图像边界傅里叶描述子
    # 输入边界数组border:（二维，(n,2)）
    # 输入重建比例proportion
    # 返回重建数组rebuild_contours：（二维，(n,2)）
    s=border.shape[0]
    rebuild_num = int(proportion * s)
    border_comp = np.empty(s, dtype=complex)
    border_comp.real = border[:, 0]
    border_comp.imag = border[:, 1]
    fourier_result = np.fft.fft(border_comp)
    magI = np.abs(fourier_result)
    indices = np.argsort(magI)
    for i in range(s - rebuild_num):
        fourier_result[indices[i]] = 0
    res = np.fft.ifft(fourier_result)
    rebuild_contours = np.zeros((s, 1, 2))
    rebuild_contours[:, 0, 0] = res[:].real
    rebuild_contours[:, 0, 1] = res[:].imag
    rebuild_contours = rebuild_contours.astype(int)
    return rebuild_contours

def location_shallow(mask):
    # 在人体二值图像中定位肩
    # 输入掩膜图像mask
    # 返回肩左、中、右点坐标数组及肩行
    dilate_img=mask/255
    nzeros_num = 0
    flag = 0
    for i in range(dilate_img.shape[0]):
        if cv.countNonZero(dilate_img[i, :]) > 0:
            nzeros_num += 1
            flag = 1
        else:
            if flag == 1:
                break
    all_w = cv.countNonZero(dilate_img)
    ave_w = int(all_w / nzeros_num)
    shallow_w = ave_w + 6
    # 遍历图像，找到双肩
    shallow_line = -1
    for i in range(mask.shape[0]):
        current_w = cv.countNonZero(mask[i, :])
        if current_w > shallow_w:
            shallow_line = i
            break
        else:
            continue
    shallow = np.zeros((3, 1, 2), dtype=np.int16)  # 存储肩左中右间点坐标
    shallow[:, 0, 1] = shallow_line
    for i in range(mask.shape[1] - 1):
        if mask[shallow_line, i] == 0 and mask[shallow_line, i + 1] == 255:
            shallow[0, 0, 0] = i + 1 + 10
        elif mask[shallow_line, i] == 255 and mask[shallow_line, i + 1] == 0:
            shallow[2, 0, 0] = i - 10
        else:
            continue
    # 肩中点横坐标默认值
    shallow[1, 0, 0] = np.int16((shallow[0, 0, 0] + shallow[2, 0, 0]) / 2)
    return shallow,shallow_line

def thinImage(src, maxIterations=-1):
    # 图像细化
    # 输入待细化图像src
    # 输出细化图像dst
    assert src.dtype == np.uint8
    height, width = src.shape
    dst = src.copy()
    count = 0
    while True:
        count += 1
        if maxIterations != -1 and count > maxIterations:
            break
        mFlag = np.ones_like(src)
        for i in range(height):
            p = dst[i, :]
            for j in range(width):
                p1 = p[j]
                if p1 != 1: continue
                p4 = (0 if j == width - 1 else p[j + 1])
                p8 = (0 if j == 0 else p[j - 1])
                p2 = (0 if i == 0 else dst[i - 1, j])
                p3 = (0 if i == 0 or j == width - 1 else dst[i - 1, j + 1])
                p9 = (0 if i == 0 or j == 0 else dst[i - 1, j - 1])
                p6 = (0 if i == height - 1 else dst[i + 1, j])
                p5 = (0 if i == height - 1 or j == width - 1 else dst[i + 1, j + 1])
                p7 = (0 if i == height - 1 or j == 0 else dst[i + 1, j - 1])
                if (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 and (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1: ap += 1
                    if p3 == 0 and p4 == 1: ap += 1
                    if p4 == 0 and p5 == 1: ap += 1
                    if p5 == 0 and p6 == 1: ap += 1
                    if p6 == 0 and p7 == 1: ap += 1
                    if p7 == 0 and p8 == 1: ap += 1
                    if p8 == 0 and p9 == 1: ap += 1
                    if p9 == 0 and p2 == 1: ap += 1
                    if ap == 1 and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                        mFlag[i, j] = 0
        dst = dst * mFlag
        for i in range(height):
            p = dst[i, :]
            for j in range(width):
                p1 = p[j]
                if p1 != 1: continue
                p4 = (0 if j == width - 1 else p[j + 1])
                p8 = (0 if j == 0 else p[j - 1])
                p2 = (0 if i == 0 else dst[i - 1, j])
                p3 = (0 if i == 0 or j == width - 1 else dst[i - 1, j + 1])
                p9 = (0 if i == 0 or j == 0 else dst[i - 1, j - 1])
                p6 = (0 if i == height - 1 else dst[i + 1, j])
                p5 = (0 if i == height - 1 or j == width - 1 else dst[i + 1, j + 1])
                p7 = (0 if i == height - 1 or j == 0 else dst[i + 1, j - 1])
                if (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 and (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1: ap += 1
                    if p3 == 0 and p4 == 1: ap += 1
                    if p4 == 0 and p5 == 1: ap += 1
                    if p5 == 0 and p6 == 1: ap += 1
                    if p6 == 0 and p7 == 1: ap += 1
                    if p7 == 0 and p8 == 1: ap += 1
                    if p8 == 0 and p9 == 1: ap += 1
                    if p9 == 0 and p2 == 1: ap += 1
                    if ap == 1 and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                        mFlag[i, j] = 0
        dst = dst * mFlag
    return dst

def filterOver(thinSrc):
    # 对骨骼化图数据进行过滤，实现两个点之间至少间隔一个空白像素
    # 输入细化后的图像thinSrc
    # 输出细化过滤后的图像thinSrc
    assert thinSrc.dtype == np.uint8
    height, width = thinSrc.shape
    for i in range(height):
        p = thinSrc[i, :]
        for j in range(width):
            p1 = p[j]
            if p1 != 1: continue
            p4 = (0 if j == width - 1 else p[j + 1])
            p8 = (0 if j == 0 else p[j - 1])
            p2 = (0 if i == 0 else thinSrc[i - 1, j])
            p3 = (0 if i == 0 or j == width - 1 else thinSrc[i - 1, j + 1])
            p9 = (0 if i == 0 or j == 0 else thinSrc[i - 1, j - 1])
            p6 = (0 if i == height - 1 else thinSrc[i + 1, j])
            p5 = (0 if i == height - 1 or j == width - 1 else thinSrc[i + 1, j + 1])
            p7 = (0 if i == height - 1 or j == 0 else thinSrc[i + 1, j - 1])
            if (p2 + p3 + p8 + p9) >= 1:
                thinSrc[i, j] = 0
    return thinSrc

def getPoints(thinSrc, raudis=4, thresholdMax=6, thresholdMin=4):
    # 从过滤后的骨骼化图像中寻找端点和交叉点
    # 输入细化过滤后的图像thinSrc
    # 输入raudis卷积半径，以当前像素点位圆心，在圆范围内判断点是否为端点或交叉点
    # 输入thresholdMax交叉点阈值，大于这个值为交叉点
    # 输入thresholdMin端点阈值，小于这个值为端点
    assert thinSrc.dtype == np.uint8
    height, width = thinSrc.shape
    tmp = thinSrc.copy()
    Point1 = []
    Point2 = []
    for i in range(height):
        for j in range(width):
            if tmp[i, j] == 0: continue
            count = 0
            for k in range(i - raudis, i + raudis + 1):
                for l in range(j - raudis, j + raudis + 1):
                    if k < 0 or l < 0 or k > height - 1 or l > width - 1:
                        continue
                    elif tmp[k, l] == 1:
                        count += 1
            if count > thresholdMax:
                Point1.append((j, i))
            if count < thresholdMin:
                Point2.append((j, i))
    return Point1, Point2

def get_shallow_mid(dst,shallow_line):
    # 查找肩中（脖颈）点横坐标
    # 输入细化过滤后的图像dst
    # 输入肩行shallow_line
    # 返回肩中（脖颈点横坐标）
    p1 = dst[shallow_line - 1, :]
    p2 = dst[shallow_line, :]
    p3 = dst[shallow_line + 1, :]
    midx = 0
    for i in range(dst.shape[1]):
        if p1[i] != 0 or p2[i] != 0 or p3[i] != 0:
            midx = i
            break
    return midx

def get_main_skel(Point,shallow):
    # 定位ROI区域左下、右下、主躯干点
    # 输入细化过滤图像的交叉点数组Point
    # 输入肩左、中、右点数组shallow
    # 输出ROI区域左下、右下、主躯干点tuple：left,right,mainP
    hit = 0
    for i in range(len(Point)):
        if Point[i][1] > shallow[0, 0, 1]+80:      # 这里的80是为了防止胸腔定位较小设置的
            left = np.array([shallow[0, 0, 0], Point[i][1]], dtype=np.int16)
            right = np.array([shallow[2, 0, 0], Point[i][1]], dtype=np.int16)
            # cv.circle(dst, Point1[i], radius, 255, -1, 8)
            hit = i
            break
    return tuple(left), tuple(right), Point[hit]

def convertTo3Channels(binImg):
    three_channel=np.zeros((binImg.shape[0],binImg.shape[1],3))
    for i in range(3):
        three_channel[:,:,i]=binImg
    return three_channel