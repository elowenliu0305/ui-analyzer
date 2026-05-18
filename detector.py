#!/usr/bin/env python3
"""页面区域检测 — 使用 v7 独立脚本的完整逻辑（不做删减）"""

import cv2
import numpy as np

COLORS = {
    "nav":     (255, 0,   0),
    "search":  (0,   255, 255),
    "content": (0,   255, 0),
    "card":    (0,   165, 255),
    "button":  (255, 0,   255),
    "input":   (255, 255, 0),
    "icon":    (255, 128, 0),
    "text":    (200, 200, 0),
    "footer":  (0,   0,   255),
    "unknown": (128, 128, 128),
}

FUNC_INFO = {
    "nav":     ("导航栏",   "点击切换"),
    "search":  ("搜索框",   "点击输入"),
    "content": ("主内容区", "滚动/点击"),
    "card":    ("内容卡片", "点击详情"),
    "button":  ("功能按钮", "点击操作"),
    "input":   ("输入框",   "点击输入"),
    "icon":    ("功能图标", "点击操作"),
    "text":    ("文本标签", "只读"),
    "footer":  ("底部栏",   "Tab 切换"),
    "unknown": ("未知区域", "可能可点"),
}


def get_content_mask(gray):
    """生成内容区域二值图"""
    mean_b = gray.mean()
    is_dark = mean_b < 100

    if is_dark:
        th = max(30, int(mean_b + gray.std() * 0.5))
        _, fg = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    else:
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return fg, is_dark


def detect_regions(img_path):
    """两阶段检测：结构检测（大核）+ 细节检测（小核/自适应阈值）"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(str(img_path))

    h, w = img.shape[:2]
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fg_mask, is_dark = get_content_mask(gray)

    all_bboxes = []

    # ─── Pass 1: 结构检测（v7 原版逻辑） ───
    if is_dark:
        structure_kernels = [5, 10, 15, 20, 30, 40]
    else:
        structure_kernels = [10, 20, 30, 50, 80]

    for ksize in structure_kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0

            if area < w * h * 0.0005:
                continue
            if area > w * h * 0.88:
                continue
            if aspect > 20 or aspect < 0.05:
                continue

            all_bboxes.append([x, y, x + bw, y + bh])

    # ─── Pass 2: 细节检测（自适应阈值 + 小核，捕捉图标和文字） ───
    # 自适应阈值能更好地捕捉低对比度文字和小图标
    binary_fine = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 5)
    if is_dark:
        binary_fine = 255 - binary_fine

    for ksize in [3, 5, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        closed = cv2.morphologyEx(binary_fine, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            if area < w * h * 0.0002:
                continue
            if area > w * h * 0.5:
                continue
            if aspect > 30 or aspect < 0.03:
                continue
            all_bboxes.append([x, y, x + bw, y + bh])

    # ─── fallback: 单独再用 Canny 边缘检测补漏 ───
    edges = cv2.Canny(gray, 30, 100)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if w * h * 0.0003 < area < w * h * 0.5 and 0.1 < aspect < 20:
            all_bboxes.append([x, y, x + bw, y + bh])

    # ─── NMS 合并 ───
    def iou(a, b):
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0

    all_bboxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    filtered = []
    for b in all_bboxes:
        keep = True
        for f in filtered:
            if iou(b, f) > 0.3:
                keep = False
                break
        if keep:
            filtered.append(b)

    # ─── 分类（v7 原版） ───
    def classify(x1, y1, x2, y2):
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / bh if bh > 0 else 0
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ratio = (bw * bh) / (w * h)

        if cy < h * 0.10 and aspect > 2:
            return "nav"
        if cy > h * 0.92 and aspect > 2:
            return "footer"
        if ratio > 0.15:
            return "content"
        if ratio < 0.008 and 0.5 < aspect < 2.0 and (cx < w * 0.08 or cx > w * 0.92):
            return "icon"
        if ratio < 0.006 and aspect > 1.5 and bh < h * 0.025:
            return "text"
        if ratio > 0.02 and 0.4 < aspect < 3:
            return "card"
        if bh < h * 0.05 and bw > w * 0.08:
            return "input"
        if ratio < 0.04 and aspect > 2 and bw > w * 0.1:
            return "search"
        if ratio < 0.008 and 0.5 < aspect < 2.0:
            return "icon"
        if ratio < 0.04 and 0.4 < aspect < 2.5:
            return "button"
        return "unknown"

    # ─── 渲染 ───
    canvas = original.copy()
    modules = []

    for i, (x1, y1, x2, y2) in enumerate(filtered):
        rtype = classify(x1, y1, x2, y2)
        desc, action = FUNC_INFO.get(rtype, ("未知", "待确认"))
        color = COLORS.get(rtype, COLORS["unknown"])

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        label = f"[{i+1}] {desc}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        lx1, ly1 = max(0, x1), max(0, y1 - th - 8)
        cv2.rectangle(canvas, (lx1, ly1), (lx1 + tw + 8, y1), color, -1)
        cv2.putText(canvas, label, (lx1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(canvas, f"> {action}", (x1 + 4, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        modules.append({
            "id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "type": rtype,
            "desc": desc,
            "action": action,
            "interactable": rtype not in ("unknown", "text"),
        })

    return canvas, modules
