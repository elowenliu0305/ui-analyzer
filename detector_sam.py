#!/usr/bin/env python3
"""SAM 精确分割检测器"""

import cv2
import numpy as np
from pathlib import Path
import torch

# 类型 → (颜色BGR, 中文名, 交互方式)
TYPE_META = {
    "nav":     ((255, 0, 0),     "导航栏",   "点击切换页面"),
    "search":  ((0, 255, 255),   "搜索框",   "点击激活 → 输入"),
    "content": ((0, 255, 0),     "内容区",   "滚动浏览"),
    "card":    ((0, 165, 255),   "内容卡片", "点击进入详情"),
    "button":  ((255, 0, 255),   "功能按钮", "点击触发操作"),
    "input":   ((255, 255, 0),   "输入框",   "点击激活 → 输入"),
    "icon":    ((255, 128, 0),   "图标",     "点击触发操作"),
    "list":    ((0, 200, 200),   "列表项",   "点击进入"),
    "avatar":  ((200, 100, 100), "头像",     "点击查看"),
    "footer":  ((0, 0, 255),     "底部栏",   "点击 Tab 切换"),
    "unknown": ((128, 128, 128), "未知区域", "可能可点击"),
}


def _classify(x1, y1, x2, y2, w, h):
    """位置+尺寸分类"""
    bw, bh = x2 - x1, y2 - y1
    aspect = bw / bh if bh > 0 else 0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ratio = (bw * bh) / (w * h)

    # 左侧图标
    if cx < w * 0.07 and ratio < 0.008:
        return "icon"
    # 顶部导航
    if cy < h * 0.10 and aspect > 2 and ratio < 0.08:
        return "nav"
    # 底部栏
    if cy > h * 0.92 and aspect > 3 and ratio < 0.05:
        return "footer"
    # 大内容区
    if ratio > 0.20:
        return "content"
    # 横向条
    if aspect > 3:
        return "search" if ratio < 0.015 else "input"
    # 中等矩形
    if 0.3 < aspect < 3.5:
        if ratio > 0.015:
            # 头像通常是小的近正方形
            if 0.7 < aspect < 1.3 and ratio < 0.01:
                return "avatar"
            return "list" if bh < h * 0.06 else "card"
    # 小方形
    if ratio < 0.015 and 0.4 < aspect < 2.5:
        if 0.7 < aspect < 1.3 and ratio < 0.008:
            return "avatar"
        return "button"
    return "unknown"


class SAMDetector:
    """使用 SAM 做自动分割"""

    def __init__(self, checkpoint_path=None):
        self.model = None
        self.mask_generator = None
        self.checkpoint_path = checkpoint_path

    def _load_model(self):
        if self.model is not None:
            return
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        ckpt = self.checkpoint_path or str(
            Path(__file__).parent / "sam_vit_b_01ec64.pth"
        )
        print(f"[SAM] 加载模型: {ckpt}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM] 设备: {device}")

        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=500,
        )
        print("[SAM] 模型加载完成")

    def detect(self, img_path):
        """检测图片中的区域，返回 (标注图BGR, 区域列表)"""
        self._load_model()

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        original = img.copy()

        print(f"[SAM] 生成 masks...")
        masks = self.mask_generator.generate(img_rgb)
        print(f"[SAM] 生成 {len(masks)} 个候选 mask")

        # ─── 合并相似的 mask ───
        def mask_iou(m1, m2):
            intersection = np.logical_and(m1["segmentation"], m2["segmentation"]).sum()
            union = np.logical_or(m1["segmentation"], m2["segmentation"]).sum()
            return intersection / union if union > 0 else 0

        # 按面积排序
        masks.sort(key=lambda m: m["area"], reverse=True)

        # NMS 合并 mask (IoU > 0.7 则合并)
        kept = []
        for m in masks:
            if any(mask_iou(m, k) > 0.7 for k in kept):
                continue
            kept.append(m)
        masks = kept
        print(f"[SAM] NMS 后: {len(masks)} 个 mask")

        # ─── 提取 bbox ───
        min_area = w * h * 0.001
        max_area = w * h * 0.85

        bboxes = []
        for m in masks:
            area = m["area"]
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = m["bbox"]
            aspect = bw / bh if bh > 0 else 0
            if aspect > 20 or aspect < 0.04:
                continue
            bboxes.append([x, y, x + bw, y + bh])

        # merge 重叠的 bbox
        def iou(a, b):
            xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
            xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            return inter / union if union > 0 else 0

        bboxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        merged = []
        for b in bboxes:
            if any(iou(b, f) > 0.35 for f in merged):
                continue
            merged.append(b)
        bboxes = merged

        # ─── 渲染 ───
        canvas = original.copy()
        modules = []

        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            rtype = _classify(x1, y1, x2, y2, w, h)
            color, cname, caction = TYPE_META.get(rtype, TYPE_META["unknown"])

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"[{i+1}] {cname}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
            tx1, ty1 = max(0, x1), max(0, y1 - th - 6)
            cv2.rectangle(canvas, (tx1, ty1), (tx1 + tw + 6, y1), color, -1)
            cv2.putText(canvas, label, (tx1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.putText(canvas, caction, (x1 + 3, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            modules.append({
                "id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "type": rtype,
                "desc": cname,
                "action": caction,
                "interactable": rtype != "unknown",
            })

        # 信息条
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0)
        type_counts = {}
        for m in modules:
            type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1
        stats = " | ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        cv2.putText(canvas, f"SAM 检测: {len(modules)} 个区域  |  {stats}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return canvas, modules


def detect_regions_sam(img_path):
    """便捷函数：使用 SAM 检测"""
    detector = SAMDetector()
    return detector.detect(img_path)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / "Desktop/截图/微信截图.png")
    print(f"检测: {path}")
    canvas, modules = detect_regions_sam(path)
    out_path = str(Path.home() / "Desktop/截图/sam_result.png")
    cv2.imwrite(out_path, canvas)
    print(f"结果保存: {out_path}")
    print(f"区域数: {len(modules)}")
    for m in modules:
        print(f"  [{m['id']:>2}] {m['type']:<8} {m['desc']:<10} → {m['action']}")
