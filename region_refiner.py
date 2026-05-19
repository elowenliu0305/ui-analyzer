#!/usr/bin/env python3
"""利用分隔线精炼区域检测结果。

核心逻辑：
  1. 把检测到的实体线作为"硬边界"——跨线的 region 必须被切开
  2. 界面线作为"软参考"——不做强制切分，但记录在 region 元信息里
"""

from detector import classify_region as _classify_region


def _split_horizontal(bbox, line_y):
    """在水平位置 line_y 处切开 bbox [x1,y1,x2,y2]，返回上、下两个 bbox。"""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, line_y], [x1, line_y + 1, x2, y2]


def _split_vertical(bbox, line_x):
    """在垂直位置 line_x 处切开 bbox，返回左、右两个 bbox。"""
    x1, y1, x2, y2 = bbox
    return [x1, y1, line_x, y2], [line_x + 1, y1, x2, y2]


def refine_regions_with_lines(regions, lines, img_w, img_h):
    """
    用分隔线精炼 region 列表。

    规则：
      - 实体线（横/竖）：如果它穿过某个 region，把该 region 切开
      - 只切宽度/高度超过 40px 的大 region（避免把小按钮也切了）
      - 切开后的子 region 根据位置重新分类 type

    返回新的 regions 列表（id 重新编号）。
    """
    # 取出横竖实体线
    h_lines = sorted([
        l for l in lines
        if l["orientation"] == "horizontal" and l["line_type"] == "实体线"
    ], key=lambda l: l["position"])

    v_lines = sorted([
        l for l in lines
        if l["orientation"] == "vertical" and l["line_type"] == "实体线"
    ], key=lambda l: l["position"])

    refined = []
    for r in regions:
        bbox = r["bbox"]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        # 小区域不切（按钮/图标/文字行）
        if bw < 40 and bh < 40:
            refined.append(r)
            continue

        # 收集穿过的线
        crossing_h = [l for l in h_lines if y1 < l["position"] < y2]
        crossing_v = [l for l in v_lines if x1 < l["position"] < x2]

        if not crossing_h and not crossing_v:
            refined.append(r)
            continue

        # 有穿越线 → 递归切分
        pieces = _split_piece(bbox, crossing_h, crossing_v)

        # 给每块重新分类
        for p in pieces:
            px1, py1, px2, py2 = p
            pw, ph = px2 - px1, py2 - py1
            if pw < 10 or ph < 10:
                continue  # 太小的碎片丢弃
            rtype = _classify_region(px1, py1, px2, py2, img_w, img_h)
            # 保留原 LLM 信息（如果有的话）
            refined.append({
                **r,
                "id": 0,
                "bbox": p,
                "type": rtype,
                "desc": rtype,
                "action": _region_action(rtype),
                "split_from": r.get("id"),
            })

    # 重编号
    for i, r in enumerate(refined):
        r["id"] = i + 1

    return refined


def _split_piece(bbox, h_lines, v_lines):
    """
    递归切分一个 bbox。

    策略：先水平切，对每一块再垂直切。
    """
    x1, y1, x2, y2 = bbox

    # 找横穿该 bbox 的水平线
    my_h = [l for l in h_lines if y1 < l["position"] < y2]
    # 找竖穿该 bbox 的垂直线
    my_v = [l for l in v_lines if x1 < l["position"] < x2]

    if not my_h and not my_v:
        return [bbox]

    # 优先水平切（页面结构一般是行分割）
    if my_h:
        # 取最靠近中间的那条
        mid_y = (y1 + y2) // 2
        best = min(my_h, key=lambda l: abs(l["position"] - mid_y))
        top, bottom = _split_horizontal(bbox, int(best["position"]))
        result = []
        # 从剩下的线中移除已用的
        remaining_h = [l for l in my_h if l != best]
        for piece in [top, bottom]:
            result.extend(_split_piece(piece, remaining_h, my_v))
        return result

    # 只有垂直线
    if my_v:
        mid_x = (x1 + x2) // 2
        best = min(my_v, key=lambda l: abs(l["position"] - mid_x))
        left, right = _split_vertical(bbox, int(best["position"]))
        remaining_v = [l for l in my_v if l != best]
        result = []
        for piece in [left, right]:
            result.extend(_split_piece(piece, my_h, remaining_v))
        return result

    return [bbox]


def _region_action(rtype):
    """根据类型返回默认交互方式。"""
    actions = {
        "nav": "点击切换", "search": "点击输入", "content": "滚动/点击",
        "card": "点击详情", "button": "点击操作", "input": "点击输入",
        "icon": "点击操作", "text": "只读", "footer": "Tab 切换",
        "list": "点击进入", "avatar": "点击查看",
    }
    return actions.get(rtype, "可能可点击")
