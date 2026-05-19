#!/usr/bin/env python3
"""线检测模块 — 识别页面中的三类分隔线：实体线、界面线（纯色/混色）、界面线（混色/混色）

算法流程：
  1. Canny 边缘检测
  2. 形态学开运算提取水平/垂直线候选
  3. 连接断裂线段 → 合并邻近线段
  4. 对每条线采样两侧像素 → 按梯度 + 方差分类
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


# ─── 调参区 ───
CANNY_LOW = 30
CANNY_HIGH = 100
HORIZ_MIN_LENGTH = 0.10   # 水平线最少占图片宽度比例
VERT_MIN_LENGTH = 0.08    # 垂直线最少占图片高度比例
MORPH_HORIZ_SIZE = 7      # 水平开运算核宽度（小核捕捉更多线，由梯度分类过滤）
MORPH_VERT_SIZE = 7       # 垂直开运算核高度
DILATE_ITER = 3           # 膨胀迭代次数（连接断裂）
GRADIENT_SOLID = 30       # 梯度幅值 ≥ 此值 → 实体线
GRADIENT_INTERFACE = 15   # 梯度幅值 ≥ 此值 → 可能是界面线
VARIANCE_THRESH = 100     # 两侧方差差异阈值（用于区分纯色/混色）
VAR_RATIO_THRESH = 3     # 方差比阈值（纯色/混色 vs 混色/混色）
COLOR_DIFF_THRESH = 25   # 颜色差异阈值（混色/混色）
MERGE_DIST_H = 5          # 水平线合并距离
MERGE_DIST_V = 5          # 垂直线合并距离


@dataclass
class DetectedLine:
    orientation: str        # "horizontal" | "vertical"
    position: int           # 主轴坐标（horiz=y, vert=x）
    start: int              # 次轴起始
    end: int                # 次轴结束
    line_type: str          # "实体线" | "界面线"
    interface_subtype: Optional[str] = None  # "纯色/混色" | "混色/混色"
    confidence: float = 0.0
    gradient_mag: float = 0.0


def sample_line_sides(img, orientation, position, start, end, side_width=4):
    """在线两侧采样，分析颜色方差。"""
    h, w = img.shape[:2]
    y = int(position) if orientation == "horizontal" else int((start + end) / 2)
    x = int(position) if orientation == "vertical" else int((start + end) / 2)

    if orientation == "horizontal":
        y_pos = int(position)
        x1, x2 = max(0, int(start)), min(w, int(end))
        if y_pos < side_width or y_pos > h - side_width or x2 <= x1:
            return 0, 0, np.zeros(3), np.zeros(3)
        side_a = img[max(0, y_pos - side_width):y_pos, x1:x2]
        side_b = img[y_pos:min(h, y_pos + side_width), x1:x2]
    else:
        x_pos = int(position)
        y1, y2 = max(0, int(start)), min(h, int(end))
        if x_pos < side_width or x_pos > w - side_width or y2 <= y1:
            return 0, 0, np.zeros(3), np.zeros(3)
        side_a = img[y1:y2, max(0, x_pos - side_width):x_pos]
        side_b = img[y1:y2, x_pos:min(w, x_pos + side_width)]

    if side_a.size == 0 or side_b.size == 0:
        return 0, 0, np.zeros(3), np.zeros(3)

    a_pixels = side_a.reshape(-1, 3).astype(np.float32)
    b_pixels = side_b.reshape(-1, 3).astype(np.float32)
    a_var = float(np.mean(np.var(a_pixels, axis=0)))
    b_var = float(np.mean(np.var(b_pixels, axis=0)))
    a_mean = np.mean(a_pixels, axis=0)
    b_mean = np.mean(b_pixels, axis=0)

    return a_var, b_var, a_mean, b_mean


def _compute_gradient(gray, orientation):
    """计算梯度——水平线用 Sobel Y，垂直线用 Sobel X。"""
    if orientation == "horizontal":
        grad = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # dy
    else:
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # dx
    return cv2.convertScaleAbs(grad)


def classify_line(gradient_mag, side_a_var, side_b_var, side_a_mean, side_b_mean):
    """三步分类：实体线 / 界面线(纯色/混色) / 界面线(混色/混色)。

    gradient_mag == -1 表示颜色过渡法检测的候选，跳过梯度检查，直接按方差+颜色分类。
    """
    if gradient_mag >= GRADIENT_SOLID:
        return "实体线", None, 0.9 - min(0.3, gradient_mag / 200)

    if gradient_mag >= 0 and gradient_mag < GRADIENT_INTERFACE:
        return None, None, 0.0

    # gradient_mag < 0 (interface 检测) 或 >= GRADIENT_INTERFACE:
    # 按两侧方差和颜色差异分类
    min_var = min(side_a_var, side_b_var)
    max_var = max(side_a_var, side_b_var)
    var_ratio = max_var / max(min_var, 1)

    # 纯色/混色：一侧明显比另一侧"干净"
    if var_ratio >= VAR_RATIO_THRESH and (max_var >= VARIANCE_THRESH or min_var < 10):
        return "界面线", "纯色/混色", 0.7

    # 混色/混色：两侧都有纹理，且颜色均值不同
    color_diff = np.linalg.norm(side_a_mean - side_b_mean)
    if min_var > 10 and max_var > 10 and color_diff > COLOR_DIFF_THRESH:
        return "界面线", "混色/混色", 0.6

    # 低频但仍可感知的线
    if color_diff > COLOR_DIFF_THRESH * 1.5:
        return "界面线", "混色/混色", 0.5

    return None, None, 0.0


def detect_lines_interface(gray_color_img):
    """
    检测界面线（非 Canny 依赖）——扫描每行/每列的局部颜色统计量变化。

    界面线的特征：
    - 两侧颜色分布明显不同
    - 但 Sobel 梯度不高（不是 hard edge）
    - 过渡在水平/垂直方向一致
    """
    # gray_color_img is passed as (gray, img)
    gray, img = gray_color_img
    h, w = gray.shape[:2]
    result = []

    # ─── 水平界面线 ───
    # 缩小宽度加速，同时提高信噪比
    sw = min(w, 800)
    if sw != w:
        small = cv2.resize(img, (sw, h), interpolation=cv2.INTER_AREA)
    else:
        small = img

    # 每行的局部均值（滑动窗口，步长 sw//8，窗口 sw//4）
    n_segments = 8
    seg_w = sw // n_segments

    row_changes = np.zeros(h - 1)
    for seg in range(n_segments):
        sx = seg * seg_w
        ex = min(sx + seg_w, sw)
        seg_colors = small[:, sx:ex, :].astype(np.float32)
        seg_mean = seg_colors.mean(axis=1).mean(axis=1)  # (h,) 灰度均值
        diff = np.abs(np.diff(seg_mean))  # (h-1,)
        # 平滑
        kernel = np.ones(5) / 5
        diff_smooth = np.convolve(np.pad(diff, 2, mode='edge'), kernel, mode='valid')
        row_changes += diff_smooth

    row_changes /= n_segments  # (h-1,)

    # 动态阈值
    thresh = max(2.0, np.percentile(row_changes, 92))

    above = row_changes > thresh
    diffs = np.diff(np.concatenate(([0], above.astype(np.int32), [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        region = row_changes[s:e]
        peak = s + np.argmax(region)
        peak_val = float(region.max())

        if peak < max(3, h * 0.01) or peak > h - max(3, h * 0.01):
            continue

        y_pos = peak + 1

        result.append(DetectedLine(
            orientation="horizontal",
            position=float(y_pos),
            start=0.0,
            end=float(w),
            line_type="",
            confidence=min(1.0, peak_val / 20),
            gradient_mag=-1,  # -1 标记：让 classify_line 走方差+颜色路径
        ))

    # ─── 垂直界面线 ───
    sh = min(h, 800)
    if sh != h:
        small_v = cv2.resize(img, (w, sh), interpolation=cv2.INTER_AREA)
    else:
        small_v = img

    n_segments_v = 8
    seg_h = sh // n_segments_v
    col_changes = np.zeros(w - 1)
    for seg in range(n_segments_v):
        sy = seg * seg_h
        ey = min(sy + seg_h, sh)
        seg_colors = small_v[sy:ey, :, :].astype(np.float32)
        seg_mean = seg_colors.mean(axis=0).mean(axis=1)  # (w,)
        diff = np.abs(np.diff(seg_mean))
        kernel = np.ones(5) / 5
        diff_smooth = np.convolve(np.pad(diff, 2, mode='edge'), kernel, mode='valid')
        col_changes += diff_smooth

    col_changes /= n_segments_v  # (w-1,)
    thresh_v = max(2.0, np.percentile(col_changes, 92))

    above = col_changes > thresh_v
    diffs = np.diff(np.concatenate(([0], above.astype(np.int32), [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        region = col_changes[s:e]
        peak = s + np.argmax(region)
        peak_val = float(region.max())

        if peak < max(3, w * 0.01) or peak > w - max(3, w * 0.01):
            continue

        x_pos = peak + 1
        grad = _compute_gradient(gray, "vertical")
        result.append(DetectedLine(
            orientation="vertical",
            position=float(x_pos),
            start=0.0,
            end=float(h),
            line_type="",
            confidence=min(1.0, peak_val / 20),
            gradient_mag=-1,
        ))

    return result


def detect_lines_morphology(gray):
    """
    用形态学方法检测水平线和垂直线。
    Canny → 水平/垂直开运算 → 提取连通域。
    """
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH, apertureSize=3)
    h, w = gray.shape[:2]

    result = []

    # ─── 水平线 ───
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_HORIZ_SIZE, 1))
    horiz_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
    # 膨胀连接断裂的线段
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    horiz_edges = cv2.dilate(horiz_edges, connect_kernel, iterations=DILATE_ITER)

    contours, _ = cv2.findContours(horiz_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < w * HORIZ_MIN_LENGTH:
            continue
        # 排除太高对象（不是线，是区域）
        if bh > h * 0.03:
            continue
        # 用原始梯度确认（水平线 → 垂直梯度 Sobel Y）
        abs_grad = _compute_gradient(gray, "horizontal")
        mean_grad = float(np.mean(abs_grad[y:y + bh, x:x + bw]))

        result.append(DetectedLine(
            orientation="horizontal",
            position=float(y + bh // 2),
            start=float(x),
            end=float(x + bw),
            line_type="",
            confidence=min(1.0, mean_grad / 100),
            gradient_mag=mean_grad,
        ))

    # ─── 垂直线 ───
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, MORPH_VERT_SIZE))
    vert_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vert_kernel)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vert_edges = cv2.dilate(vert_edges, connect_kernel, iterations=DILATE_ITER)

    contours, _ = cv2.findContours(vert_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < h * VERT_MIN_LENGTH:
            continue
        if bw > w * 0.03:
            continue

        abs_grad = _compute_gradient(gray, "vertical")
        mean_grad = float(np.mean(abs_grad[y:y + bh, x:x + bw]))

        result.append(DetectedLine(
            orientation="vertical",
            position=float(x + bw // 2),
            start=float(y),
            end=float(y + bh),
            line_type="",
            confidence=min(1.0, mean_grad / 100),
            gradient_mag=mean_grad,
        ))

    return result


def merge_nearby_lines(lines):
    """合并位置相近、方向相同的线，保留最长的跨度。"""
    if not lines:
        return []
    horiz = sorted([l for l in lines if l.orientation == "horizontal"],
                   key=lambda l: l.position)
    vert = sorted([l for l in lines if l.orientation == "vertical"],
                  key=lambda l: l.position)

    def merge_group(group, max_dist):
        if not group:
            return []
        merged = [group[0]]
        for l in group[1:]:
            prev = merged[-1]
            if abs(l.position - prev.position) <= max_dist:
                prev.start = min(prev.start, l.start)
                prev.end = max(prev.end, l.end)
                if l.confidence > prev.confidence:
                    prev.line_type = l.line_type
                    prev.interface_subtype = l.interface_subtype
                    prev.confidence = l.confidence
                    prev.gradient_mag = l.gradient_mag
            else:
                merged.append(l)
        return merged

    return merge_group(horiz, MERGE_DIST_H) + merge_group(vert, MERGE_DIST_V)


def detect_lines_column_projection(gray):
    """
    用 Sobel X 列投影检测垂直线——捕捉形态学和颜色统计都遗漏的列边界。

    算法：
      1. Sobel X 算子得到垂直边缘响应
      2. 阈值化保留强边缘
      3. 逐列求和得到投影曲线
      4. 平滑后找波峰
      5. 波峰位置 > 阈值 → 垂直线候选

    适合检测两类被现有方法遗漏的竖线：
      - Canny 无法捕捉的弱边缘（细边框、浅色分割线）
      - 颜色统计无法捕捉的薄结构边界（两侧颜色相近但存在细线分隔）
    """
    # Sobel X（垂直边缘）
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobelx_abs = cv2.convertScaleAbs(sobelx)

    # 阈值化
    _, strong = cv2.threshold(sobelx_abs, 35, 255, cv2.THRESH_BINARY)

    h, w = gray.shape[:2]

    # 逐列求和
    col_sum = np.sum(strong, axis=0).astype(np.float32)  # (w,)

    # 平滑
    kernel = np.ones(7) / 7
    col_smooth = np.convolve(col_sum, kernel, mode='same')

    # 动态阈值：取百分位数
    thresh = max(100, np.percentile(col_smooth, 98))

    result = []
    in_peak = False
    peak_max = 0
    peak_pos = 0

    for x in range(1, w - 1):
        if col_smooth[x] > thresh:
            if not in_peak:
                in_peak = True
                peak_max = col_smooth[x]
                peak_pos = x
            elif col_smooth[x] > peak_max:
                peak_max = col_smooth[x]
                peak_pos = x
        else:
            if in_peak:
                in_peak = False
                # 取波峰最大值位置（不限制宽度，宽波峰仍取最强列）
                grad = float(np.max(sobelx_abs[:, peak_pos]))
                if grad > 0:
                    result.append(DetectedLine(
                        orientation="vertical",
                        position=float(peak_pos),
                        start=0.0,
                        end=float(h),
                        line_type="",
                        confidence=min(0.8, peak_max / thresh / 2),
                        gradient_mag=grad,
                    ))

    return result


def detect_lines(img_path):
    """
    主入口：检测图片中的所有分隔线。

    返回:
        lines: list[dict] — 每条线含 orientation, position, start, end,
                           line_type, interface_subtype, confidence
        line_vis_img: np.ndarray — BGR 标注图
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(str(img_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # ─── 第一阶段：形态学法检测实体线候选 ───
    candidates = detect_lines_morphology(gray)

    # ─── 第二阶段：颜色统计法检测界面线候选 ───
    candidates += detect_lines_interface((gray, img))

    # ─── 第三阶段：Sobel 列投影检测竖线候选（补漏：细边框、薄分割线） ───
    candidates += detect_lines_column_projection(gray)

    # ─── 合并邻近线 ───
    candidates = merge_nearby_lines(candidates)

    # ─── 分类每条线 ───
    classified = []
    for l in candidates:
        a_var, b_var, a_mean, b_mean = sample_line_sides(
            img, l.orientation, l.position, l.start, l.end)
        line_type, subtype, conf = classify_line(
            l.gradient_mag, a_var, b_var, a_mean, b_mean)
        if line_type is None:
            continue
        l.line_type = line_type
        l.interface_subtype = subtype
        l.confidence = conf
        classified.append(l)

    # ─── 生成可视化 ───
    vis = draw_lines(img, classified)

    return [asdict(l) for l in classified], vis


# ─── 可视化 ───
LINE_COLORS = {
    "实体线": (0, 0, 255),
    "界面线": {
        "纯色/混色": (255, 165, 0),
        "混色/混色": (255, 255, 0),
        None: (0, 255, 255),
    },
}


def draw_lines(img, lines):
    """在图片上绘制检测到的分隔线。"""
    canvas = img.copy()
    h, w = img.shape[:2]

    for l in lines:
        if l.line_type == "实体线":
            color = LINE_COLORS["实体线"]
        else:
            color = LINE_COLORS["界面线"].get(l.interface_subtype, (0, 255, 255))

        if l.orientation == "horizontal":
            pt1 = (int(l.start), int(l.position))
            pt2 = (int(l.end), int(l.position))
        else:
            pt1 = (int(l.position), int(l.start))
            pt2 = (int(l.position), int(l.end))

        thickness = 2 if l.line_type == "实体线" else 1
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

        # 标注
        mx = (pt1[0] + pt2[0]) // 2
        my = (pt1[1] + pt2[1]) // 2
        label = l.line_type
        if l.interface_subtype:
            label += f"({l.interface_subtype})"
        cv2.putText(canvas, label,
                    (min(mx + 4, w - 120), min(my - 4, h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # 信息栏
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 26), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
    line_counts = {}
    for l in lines:
        key = l.line_type
        if l.interface_subtype:
            key += f"/{l.interface_subtype}"
        line_counts[key] = line_counts.get(key, 0) + 1
    stats = " | ".join(f"{k}:{v}" for k, v in line_counts.items())
    cv2.putText(canvas, f"线检测: {len(lines)} 条  |  {stats}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return canvas


if __name__ == "__main__":
    import sys
    from pathlib import Path
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / "Desktop/截图/微信截图.png")
    print(f"线检测: {path}")
    lines, vis = detect_lines(path)
    print(f"检测到 {len(lines)} 条线:")
    for l in lines:
        print(f"  {l['orientation'][:4]:>4} pos={int(l['position']):>4} {int(l['start']):>4}-{int(l['end']):<4}  "
              f"{l['line_type']}  {l.get('interface_subtype','') or '-'}  conf={l['confidence']:.2f}")
    out_path = str(Path.home() / "Desktop/截图/lines_result.png")
    cv2.imwrite(out_path, vis)
    print(f"可视化: {out_path}")
