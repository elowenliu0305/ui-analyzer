#!/usr/bin/env python3
"""LLM 理解模块 — 用 DeepSeek (OpenAI 兼容) 分析每个区域的功能和交互方式"""

import base64
import json
import re
from openai import OpenAI


def _encode_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def analyze_regions_with_llm(config, original_img_path, regions):
    """
    将整张标注图发给 LLM，让 LLM 同时理解所有区域。

    参数:
      config: dict — 包含 llm_api_key, llm_base_url, llm_model
      original_img_path: str — 原图路径
      regions: list[dict] — 区域列表

    返回: list[dict] — 每个区域补充了 llm_desc 和 llm_action
    """
    api_key = config.get("llm_api_key", "")
    if not api_key:
        return regions

    base_url = config.get("llm_base_url", "https://api.deepseek.com")
    model = config.get("llm_model", "deepseek-chat")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 把区域信息传给 LLM
    regions_info = []
    for r in regions:
        regions_info.append({
            "id": r["id"],
            "bbox": r["bbox"],
            "cv_type": r["type"],
        })

    b64 = _encode_image(original_img_path)

    prompt = f"""你是一个 UI 分析师。下面是一张页面截图，上面用不同颜色的矩形框和编号标注了功能区域。

请分析每个编号区域，判断它是什么功能、有什么用途、用户应该如何交互。

以 JSON 格式返回，格式为数组：
[
  {{
    "id": 编号,
    "llm_desc": "简短的功能描述（中文，10字以内）",
    "llm_action": "交互方式（中文，10字以内，如：点击输入、点击跳转、滚动浏览）"
  }},
  ...
]

只返回 JSON 数组，不要其他内容。必须包含所有 {len(regions_info)} 个区域。

已知 OpenCV 初步分类结果（供参考，不一定准确）：
{json.dumps(regions_info, ensure_ascii=False, indent=2)}
"""

    # 尝试图片理解；若模型不支持视觉则 fallback 为纯文本
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=4096,
            temperature=0.1,
        )
        text = response.choices[0].message.content
    except Exception as vision_err:
        print(f"[LLM] 图片理解失败，尝试纯文本: {vision_err}")
        # fallback: 纯文本，不带图片
        text_prompt = prompt + f"\n\n由于我无法看到图片，请根据 OpenCV 的分类结果（cv_type）和坐标位置推测每个区域的功能。bbox 格式为 [x1, y1, x2, y2]。"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text_prompt}],
            max_tokens=4096,
            temperature=0.1,
        )
        text = response.choices[0].message.content

    # 提取 JSON
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            result = json.loads(match.group(1))
        else:
            match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                print(f"[LLM] 解析失败，原始响应:\n{text[:500]}")
                raise ValueError("无法从 LLM 响应中解析 JSON")

    # 合并回 regions
    llm_map = {r["id"]: r for r in result}
    for r in regions:
        llm_data = llm_map.get(r["id"], {})
        r["llm_desc"] = llm_data.get("llm_desc", r.get("type", "未知"))
        r["llm_action"] = llm_data.get("llm_action", r.get("action", "可能可点击"))

    return regions
