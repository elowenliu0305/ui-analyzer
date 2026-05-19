#!/usr/bin/env python3
"""UI Analyzer Web App — Flask 后端"""

import io
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory

from detector import detect_regions
from detector_sam import SAMDetector
from llm_analyzer import analyze_regions_with_llm
from line_detector import detect_lines as detect_lines_in_image
from region_refiner import refine_regions_with_lines

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
OUTPUT = BASE / "output"
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

# ─── 配置 ───
CONFIG_PATH = BASE / "config.json"
_config = None


def load_config():
    global _config
    if _config is None:
        try:
            _config = json.load(open(CONFIG_PATH))
        except Exception:
            _config = {"llm_api_key": ""}
    return _config


# SAM 检测器（懒加载）
_sam_detector = None


def _get_nearby_lines(bbox, lines, max_dist=20):
    """找离 bbox [x1,y1,x2,y2] 最近的线。"""
    x1, y1, x2, y2 = bbox
    nearby = []
    for l in lines:
        if l["orientation"] == "horizontal":
            pos = l["position"]
            lx1, lx2 = l["start"], l["end"]
            # 线在 bbox 附近且水平跨度与 bbox 有重叠
            if y1 - max_dist <= pos <= y2 + max_dist and lx1 < x2 and lx2 > x1:
                dist = min(abs(pos - y1), abs(pos - y2))
                nearby.append({**l, "distance": round(dist, 1)})
        else:
            pos = l["position"]
            ly1, ly2 = l["start"], l["end"]
            if x1 - max_dist <= pos <= x2 + max_dist and ly1 < y2 and ly2 > y1:
                dist = min(abs(pos - x1), abs(pos - x2))
                nearby.append({**l, "distance": round(dist, 1)})
    nearby.sort(key=lambda x: x["distance"])
    return nearby[:3]


def get_sam_detector():
    global _sam_detector
    if _sam_detector is None:
        print("[APP] 初始化 SAM 检测器...")
        _sam_detector = SAMDetector()
    return _sam_detector


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def api_config():
    """返回 LLM 是否已配置（不暴露 key）"""
    cfg = load_config()
    return jsonify({"llm_configured": bool(cfg.get("llm_api_key"))})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """上传图片 → 检测 → LLM 理解 → 返回标注图 + 区域列表"""
    if "image" not in request.files:
        print(f"[ERROR] 请求中没有 image 字段, keys={list(request.files.keys())}")
        return jsonify({"error": "请上传图片（字段名: image）"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "请选择图片"}), 400

    mode = request.form.get("mode", "opencv")
    print(f"[UPLOAD] {file.filename} ({file.content_length or 'unknown'} bytes, mode={mode})")

    # 保存原图
    ext = Path(file.filename).suffix or ".png"
    uid = uuid.uuid4().hex[:8]
    img_name = f"{uid}{ext}"
    img_path = UPLOADS / img_name
    file.save(str(img_path))

    try:
        if mode == "sam":
            detector = get_sam_detector()
            annotated, regions = detector.detect(str(img_path))
        else:
            annotated, regions = detect_regions(str(img_path))
        print(f"[DETECT] mode={mode}, {len(regions)} 个区域")
    except Exception as e:
        print(f"[ERROR] 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"检测失败: {str(e)}"}), 500

    # ─── 线检测（始终进行） ───
    try:
        lines, line_vis = detect_lines_in_image(str(img_path))
        print(f"[LINE] 检测到 {len(lines)} 条分隔线")
    except Exception as e:
        print(f"[LINE] 检测失败: {e}")
        lines, line_vis = [], None

    # ─── 用线精炼区域（只精炼数据，标注图保持 OpenCV 原样以避免杂乱） ───
    if lines:
        try:
            pre_count = len(regions)
            img_h, img_w = annotated.shape[:2]
            regions = refine_regions_with_lines(regions, lines, img_w, img_h)
            print(f"[REFINE] 区域: {pre_count} → {len(regions)}（用实体线切分+合并）")
        except Exception as e:
            print(f"[REFINE] 精炼失败: {e}")

    # ─── 自动 LLM 理解（精炼之后，分析的是最终区域） ───
    cfg = load_config()
    if cfg.get("llm_api_key"):
        try:
            regions = analyze_regions_with_llm(cfg, str(img_path), regions)
            print(f"[LLM] 理解完成: {len(regions)} 个区域")
        except Exception as e:
            print(f"[LLM] 理解失败: {e}")

    # 保存标注图
    annot_name = f"{uid}_annotated.png"
    annot_path = OUTPUT / annot_name
    cv2.imwrite(str(annot_path), annotated)

    # 保存线检测可视化
    line_vis_name = None
    if line_vis is not None:
        line_vis_name = f"{uid}_lines.png"
        line_vis_path = OUTPUT / line_vis_name
        cv2.imwrite(str(line_vis_path), line_vis)

    # 合并线信息到 regions
    for r in regions:
        r["nearby_lines"] = _get_nearby_lines(r["bbox"], lines)

    return jsonify({
        "image": f"/uploads/{img_name}",
        "annotated": f"/output/{annot_name}",
        "lines_image": f"/output/{line_vis_name}" if line_vis_name else None,
        "regions": regions,
        "lines": lines,
        "mode": mode,
    })


# ─── 静态文件服务 ───
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory("uploads", filename)


@app.route("/output/<path:filename>")
def output(filename):
    return send_from_directory("output", filename)


if __name__ == "__main__":
    PORT = 8080
    cfg = load_config()
    if cfg.get("llm_api_key"):
        print(f"🔑 LLM 已配置（{cfg.get('llm_provider','deepseek')}），检测后将自动分析功能")
    else:
        print("ℹ️  LLM 未配置，请在 config.json 中填入 llm_api_key")
    print(f"🚀 UI Analyzer 启动: http://localhost:{PORT}")
    app.run(debug=True, port=PORT)
