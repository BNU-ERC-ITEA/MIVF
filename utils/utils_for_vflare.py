import numpy as np
import math
import cv2
import copy


"""
read the source light position and generate the gaussian mask
"""
def calculate_angle(mask_size, point_pos):
    """
    根据传入的坐标点计算角度
    :param mask_size: 模板大小
    :param point_pos: 坐标位置
    :return: angel_deg: 角度
    """
    # 计算画面中心点
    center_x = mask_size[1] / 2
    center_y = mask_size[0] / 2

    # 计算从中心点到点A的向量
    dx = point_pos[1] - center_x
    dy = point_pos[0] - center_y

    # 使用 atan2 计算角度（弧度）
    angle_rad = math.atan2(dy, dx)

    # 将弧度转换为度数
    angle_deg = math.degrees(angle_rad)

    # 调整角度范围，使其以宽度方向为正方向
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


def calculate_distance(mask_size, point_pos):
    """
    根据传入的坐标点计算距离
    :param mask_size: 模板大小
    :param point_pos: 坐标位置
    :return: distance: 距离
    """
    distance = np.sqrt((point_pos[0] - mask_size[0] / 2) ** 2 + (point_pos[1] - mask_size[1] / 2) ** 2)
    return distance


def generate_anisotropic_gaussian_template(image_shape, sigma_x, sigma_y, angle):
    """
    生成一个与图像大小一致的各向异性的高斯模板。
    :param image_shape: 图像的形状 (height, width)
    :param sigma_x: x 轴的标准差
    :param sigma_y: y 轴的标准差
    :param angle: 主轴方向的角度（单位为度）
    :return: 各向异性的高斯模板
    """
    height, width = image_shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = width // 2, height // 2
    x_rel = x - center_x  # 计算相对中心点的坐标
    y_rel = y - center_y
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    x_rot = x_rel * cos_theta + y_rel * sin_theta
    y_rot = -x_rel * sin_theta + y_rel * cos_theta
    template = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
    template /= np.max(template)  # 归一化

    return template


"""
automatic generate the mask from two frame
"""
def top_hat(image,kernel_size,inter):
    # 顶帽：原图减去开运算后的图：src - opening
    img = image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))  # 定义结构元素
    image_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=inter)
    return image_tophat


def black_hat(image):
    # 顶帽：原图减去开运算后的图：src - opening
    img = image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))  # 定义结构元素
    image_blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return image_blackhat


def erode(image, k_size):
    """
    腐蚀操作
    :param image: 输入图像
    :param k_size: 腐蚀核算子
    :return:
    """
    # kernel = np.ones((k_size, k_size), np.uint8)  # 矩形结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))  # 定义结构元素
    erosion = cv2.erode(image, kernel)  # 腐蚀
    return erosion


def detect_sift(img):
    """
    SIFT特征检测
    :param img:输入图像
    :return: 特征点和特征向量
    """
    sift = cv2.SIFT_create()  # SIFT特征提取对象
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    kp = sift.detect(gray, None)  # 关键点位置
    kp, des = sift.compute(gray, kp)  # des为特征向量
    return kp, des


def RANSAC(img1, img2, kp1, kp2, matches, display=False):
    """
    RANSAC算法,进一步筛选匹配的特征点
    :param img1: 输入的左图
    :param img2: 输入的右图
    :param kp1: 左图特征点
    :param kp2: 右图特征点
    :param matches: 经过暴力匹配或者knn匹配的match对
    :param display: 是否显示优化前后的匹配结果
    :return: 从左图变换到右图的单应性变换矩阵H
    """
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    matchType = type(matches[0])
    good = []
    M = None
    if isinstance(matches[0], cv2.DMatch):
        # 搜索使用的是match
        good = matches
    else:
        # 搜索使用的是knnMatch
        for m, n in matches:
            if m.distance < 0.01 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M: 3x3 变换矩阵.应该写H的，但是由于核外边变量重名了
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
        matchesMask = mask.ravel().tolist()

        # h, w = img1.shape
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    if display:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        draw_params1 = dict(matchColor=(0, 255, 0),  # draw matches in green color
                            singlePointColor=None,
                            matchesMask=None,
                            flags=2)
        img33 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params1)

        cv2.namedWindow("before", cv2.WINDOW_NORMAL)
        cv2.namedWindow("now", cv2.WINDOW_NORMAL)
        cv2.imshow("before", img33)
        cv2.imshow("now", img3)
        cv2.waitKey(0)

    return M


def flann_match(img1, img2, kp1, kp2, des1, des2, display=False):
    """
    （1）FLANN匹配器
    :param img1: 匹配图像1
    :param img2: 匹配图像2
    :param kp1: 匹配图像1的特征点
    :param kp2: 匹配图像2的特征点
    :param des1: 匹配图像1的描述子
    :param des2: 匹配图像2的描述子
    :return: 匹配结果matches
    """
    # SIFT方法
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        trees=5)
    search_params = dict(check=50)

    # 定义FLANN参数
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.match(des1, des2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if display:
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.imshow("Matches", img3)
        cv2.waitKey(0)
    return matches


def bf_match(img1, img2, kp1, kp2, des1, des2, display=False):
    """
    BF暴力匹配
    :param img1: 输入的左图
    :param img2: 输入的右图
    :param kp1: 左图特征点
    :param kp2: 右图特征点
    :param des1: 左图特征向量
    :param des2: 右图特征向量
    :param display: 是否显示匹配后的结果
    :return: None
    """
    bf = cv2.BFMatcher(crossCheck=True)  # 匹配对象
    matches = bf.match(des1, des2)  # 进行两个特征矩阵的匹配
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)  # 绘制匹配结果
    if display:
        cv2.namedWindow('res', cv2.WINDOW_NORMAL)
        cv2.imshow('res', res)
        cv2.waitKey(0)


def calculate_line_brightness(image, angle, line_width):
    """
    计算所设定角度的、大小直线的亮度，默认直线过图像中心，这是为了计算眩光和光源所在的直线
    :param image: 输入图像
    :param angle: 角度
    :param line_width: 直线宽度
    :return: average_brightness: 这条直线的平均亮度
             mask: 用于标记直线上的像素，用于后续绘制直线
    """
    # 获取图像的中心点
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # 创建一个掩码，用于标记直线上的像素
    mask = np.zeros_like(image, dtype=np.uint8)

    # 计算直线的斜率和截距
    theta = np.radians(angle)
    m = np.tan(theta)
    c = center_y - m * center_x

    # 计算直线与图像边界的交点
    def find_intersection(m, c, x1, x2, y1, y2):
        intersections = []
        if m != 0:
            # 与左右边界相交
            y_left = m * x1 + c
            y_right = m * x2 + c
            if y1 <= y_left <= y2:
                intersections.append((x1, int(y_left)))
            if y1 <= y_right <= y2:
                intersections.append((x2, int(y_right)))
        else:
            # 水平直线
            intersections.append((x1, int(c)))
            intersections.append((x2, int(c)))

        if m != float('inf'):
            # 与上下边界相交
            x_top = (y1 - c) / m
            x_bottom = (y2 - c) / m
            if x1 <= x_top <= x2:
                intersections.append((int(x_top), y1))
            if x1 <= x_bottom <= x2:
                intersections.append((int(x_bottom), y2))
        else:
            # 垂直直线
            intersections.append((int(c), y1))
            intersections.append((int(c), y2))

        return intersections

    def find_intersection_dev(m, c, x1, x2, y1, y2):
        intersections = []
        if m != 0 and m != float('inf'):
            # 与左右边界相交
            y_left = m * x1 + c
            y_right = m * x2 + c
            if y1 <= y_left <= y2:
                intersections.append((x1, int(y_left)))
            if y1 <= y_right <= y2:
                intersections.append((x2, int(y_right)))
            # 与上下边界相交
            x_top = (y1 - c) / m
            x_bottom = (y2 - c) / m
            if x1 <= x_top <= x2:
                intersections.append((int(x_top), y1))
            if x1 <= x_bottom <= x2:
                intersections.append((int(x_bottom), y2))
        elif m == 0:
            # 水平直线
            intersections.append((x1, int(c)))
            intersections.append((x2, int(c)))
        else:
            # 垂直直线
            intersections.append((int(c), y1))
            intersections.append((int(c), y2))

        return intersections

    # intersections = find_intersection(m, c, 0, width - 1, 0, height - 1)
    intersections_dev = find_intersection_dev(m, c, 0, width - 1, 0, height - 1)
    # if len(intersections) < 2:
    #     raise ValueError("无法找到有效的交点")
    if len(intersections_dev) < 2:
        raise ValueError("无法找到有效的交点")
    if len(intersections_dev) > 2:
        print('此角度下的直线穿过对角线，去除后两个元素')
        intersections_dev = intersections_dev[:-2]

    # 选择最近的两个交点作为起始点和结束点
    x1, y1 = intersections_dev[0]
    x2, y2 = intersections_dev[1]

    # 使用 Bresenham 算法绘制直线
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    # 在掩码上标记直线上的像素
    for point in points:
        x, y = point
        for i in range(-line_width // 2, line_width // 2 + 1):
            for j in range(-line_width // 2, line_width // 2 + 1):
                if 0 <= x + i < width and 0 <= y + j < height:
                    mask[y + j, x + i] = 255

    # 计算直线上的像素亮度总和
    brightness_sum = np.sum(image[mask > 0])

    # 计算直线上的像素数量
    pixel_count = np.count_nonzero(mask)

    # 计算平均亮度
    average_brightness = brightness_sum / pixel_count if pixel_count > 0 else 0

    return average_brightness, mask, intersections_dev


def most_bright_line(image, line_width):
    """
    计算最亮的直线,返回这条直线的亮度和掩码
    :param image: 输入图像
    :param line_width: 定义直线的宽度，越宽扫描的像素越多
    :return: max_brightness_angle: 最亮的直线的角度
            max_brightness_line_mask: 最亮的直线的掩码,用于画图，一般不需要
    """
    img = image
    l_width = line_width

    brightness_value = []  # 记录每一条直线的亮度和掩码
    max_brightness = 0
    max_brightness_angle = 0
    max_brightness_line_mask = None
    max_brightness_intersections = None
    for i in range(0, 180):  # 这部分算法写的有点麻烦了，直接通过lamda排序后取前1也可以，这里不想改了
        br_score, mask, intersections_points = calculate_line_brightness(img, i, l_width)
        brightness_value.append((i, br_score, mask, intersections_points))
        if br_score > max_brightness:
            max_brightness = br_score
            max_brightness_angle = i
            max_brightness_line_mask = mask
            max_brightness_intersections = intersections_points

    # 找出最亮的5条直线及其角度
    top_8_brightness_data = sorted(brightness_value, key=lambda x: x[1], reverse=True)[:8]
    top_8_brightness_data_temp = top_8_brightness_data  # 测试用，暂时
    top_8_brightness_data = [(x[0], x[1]) for x in top_8_brightness_data]
    top_8_average_brightness = sum(x[1] for x in top_8_brightness_data) / len(top_8_brightness_data)
    # 置信规则，如果与最大亮度角度不超过20度的角度有4条，那么认为此是眩光直线。
    differ_angle = []
    for angle, score in top_8_brightness_data:
        if angle < max_brightness_angle:
            differ_angle.append((min((max_brightness_angle - angle), (angle + 180 - max_brightness_angle))) % 90)
        else:
            differ_angle.append((min((angle - max_brightness_angle), (max_brightness_angle + 180 - angle))) % 90)
    count = sum(1 for x in differ_angle if abs(x) < 20)
    if count >= 5:
        max_brightness_line_mask = top_8_brightness_data_temp[4][2]
        max_brightness_angle = top_8_brightness_data_temp[4][0]
        max_brightness_intersections = top_8_brightness_data_temp[4][3]
        max_brightness_line_mask = np.stack([max_brightness_line_mask] * 3, axis=2)
        return max_brightness_angle, max_brightness_line_mask, max_brightness_intersections
    else:
        max_brightness_line_mask = np.stack([max_brightness_line_mask] * 3, axis=2)
        return None, max_brightness_line_mask, max_brightness_intersections
        # return None, None, None, None


def flare_mask(img1, img2):
    """
    通过两帧，通过sift特征对齐、帧差、滤波、腐蚀得到近似眩光模板，并确定眩光所在直线。返回眩光模板和眩光所在直线。
    :param img1:
    :param img2:
    :return:
    """
    kp1, des1 = detect_sift(img1)
    kp2, des2 = detect_sift(img2)
    # 匹配
    # bf_match(img1, img2, kp1, kp2, des1, des2, False)
    f_match = flann_match(img2, img1, kp2, kp1, des2, des1, False)
    H = RANSAC(img2, img1, kp2, kp1, f_match, False)
    # 对齐
    img2_warp = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    mask_img2_warp = copy.deepcopy(img2_warp)
    mask_img2_warp_gray = cv2.cvtColor(mask_img2_warp, cv2.COLOR_BGR2GRAY)
    img1_copy = copy.deepcopy(img1)
    mask_img2_warp_gray[mask_img2_warp_gray > 0] = 1
    mask_img2_warp_gray[mask_img2_warp_gray <= 0] = 0
    mask_img2_warp = np.stack([mask_img2_warp_gray] * 3, axis=2)
    img1_cut = img1_copy * mask_img2_warp
    # 帧减
    # residual = np.uint8(np.float32(img1_cut) - np.float32(img2_warp))  # 直接的减法不太行
    img1_cut = cv2.GaussianBlur(img1_cut, (21, 21), 0)  # 在低频域上帧减
    from matplotlib import pyplot as plt
    plt.imshow(img1_cut)
    plt.show()
    img2_warp = cv2.GaussianBlur(img2_warp, (21, 21), 0)
    plt.imshow(img1_cut)
    plt.show()
    residual_cv = cv2.subtract(img1_cut, img2_warp)
    plt.imshow(residual_cv)
    plt.show()
    # 滤波
    median_r = cv2.medianBlur(residual_cv, 11)
    gauss_r_mf = cv2.GaussianBlur(median_r, (11, 11), 0)
    # 腐蚀残差图
    erode_flare = erode(gauss_r_mf, 11)
    gray_erode_flare = cv2.cvtColor(erode_flare, cv2.COLOR_BGR2GRAY)
    from matplotlib import pyplot as plt
    plt.imshow(gray_erode_flare, cmap='gray')
    plt.show()
    # 找到所在直线的位置
    best_angle, mask_line, points = most_bright_line(gray_erode_flare, 15)
    # 根据返回角度生成高斯模板
    mask_shape = img1.shape[0:2]
    if best_angle is not None:
        mask_line_length = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
        sigma_warp_axis = mask_line_length / 6
        sigma_warp_vertical_axis = sigma_warp_axis * mask_shape[0] / mask_shape[1]
        # print(mask_line_length, sigma_warp_axis, sigma_warp_vertical_axis)
        template = generate_anisotropic_gaussian_template(mask_shape, sigma_warp_axis, sigma_warp_vertical_axis, best_angle)
    else:
        template = np.ones(mask_shape, dtype=np.float32)

    return template


