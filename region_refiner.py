#!/usr/bin/env python3
"""利用分隔线精炼区域检测结果。

核心逻辑：
  1. 把检测到的实体线作为"硬边界"——跨线的 region 必须被切开
  2. 界面线作为"软参考"——不做强制切分，但记录在 region 元信息里
  3. 切完后合并相邻的同类型小区域（没被线隔开的两行标题）
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


def _has_line_between(y1, y2, lines):
    """检查 y1 和 y2 之间是否有线（实体线或界面线）。"""
    for l in lines:
        if l["orientation"] != "horizontal":
            continue
        pos = l["position"]
        if y1 + 2 < pos < y2 - 2:  # 留 2px 余量，避免紧贴的误判
            return True
    return False


def _merge_adjacent_regions(regions, lines):
    """
    合并相邻的同类型小区域（如两行标题）。

    规则：
      - 两个 region 类型相同（或都是 text）
      - 垂直相邻（gap < 2 倍矮者高度）
      - 水平重叠 > 50%
      - 之间没有线穿过
    """
    if len(regions) < 2:
        return regions

    # 按 y 排序
    sorted_r = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    merged = True
    while merged:
        merged = False
        new_list = []
        used = set()

        for i, a in enumerate(sorted_r):
            if i in used:
                continue
            x1a, y1a, x2a, y2a = a["bbox"]

            for j, b in enumerate(sorted_r):
                if j <= i or j in used:
                    continue
                x1b, y1b, x2b, y2b = b["bbox"]

                # 必须有水平重叠 > 50%
                overlap = max(0, min(x2a, x2b) - max(x1a, x1b))
                width_a = x2a - x1a
                width_b = x2b - x1b
                min_w = min(width_a, width_b)
                if min_w <= 0 or overlap / min_w < 0.5:
                    continue

                # 类型必须兼容
                type_a = a.get("type", "")
                type_b = b.get("type", "")
                # text + text 合并；nav + nav 合并；text + nav 不合并
                mergeable_types = {"text", "nav", "footer", "input", "search", "card", "button"}
                if type_a not in mergeable_types or type_b not in mergeable_types:
                    continue
                if type_a != type_b and not (type_a in ("text", "nav") and type_b in ("text", "nav")):
                    continue

                # 判断谁上谁下
                top_r, bot_r = (a, b) if y1a <= y1b else (b, a)
                _, ty1, _, ty2 = top_r["bbox"]
                _, by1, _, by2 = bot_r["bbox"]

                # 必须有垂直相邻关系：上者的底到下者的顶
                gap = by1 - ty2
                top_h = ty2 - ty1
                bot_h = by2 - by1

                if gap < -5:  # 重叠过多，不是相邻区域
                    continue
                if gap > max(top_h, bot_h) * 2:  # 间隔太大
                    continue

                # 之间没有线
                if _has_line_between(ty2, by1, lines):
                    continue

                # 合并!
                new_bbox = [
                    min(x1a, x1b),
                    min(y1a, y1b),
                    max(x2a, x2b),
                    max(y2a, y2b),
                ]
                # 新区域
                merged_r = {
                    **a,
                    "bbox": new_bbox,
                    "type": type_a if type_a != "nav" else type_a,
                    "desc": type_a if type_a != "nav" else type_a,
                    "action": _region_action(type_a),
                    "merged_from": [a.get("id", 0), b.get("id", 0)],
                }
                new_list.append(merged_r)
                used.add(i)
                used.add(j)
                merged = True
                break  # 每次只合并一对，然后重新扫描

            if i not in used:
                new_list.append(a)

        sorted_r = sorted(new_list, key=lambda r: r["bbox"][1])

    return sorted_r


def refine_regions_with_lines(regions, lines, img_w, img_h, return_split_vis=False):
    """
    用分隔线精炼 region 列表。

    规则：
      - 实体线（横/竖）：如果它穿过某个 region，把该 region 切开
      - 只切宽度/高度超过 40px 的大 region（避免把小按钮也切了）
      - 切开后的子 region 根据位置重新分类 type
      - 切完后合并相邻同类型小区域（无线间隔的两行标题等）

    参数:
        return_split_vis: 如果为 True，额外返回 (split_regions, split_pieces)
                          用于生成切割步骤可视化

    返回:
        默认返回新的 regions 列表（id 重新编号）
        如果 return_split_vis=True，返回 (regions, split_regions, split_pieces)
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
    split_regions = []   # 被切的原始 region（切割步骤可视化用）
    split_pieces = []    # 切出来的碎片 bbox（切割步骤可视化用）
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

        if return_split_vis:
            split_regions.append(r)
            for p in pieces:
                px1, py1, px2, py2 = p
                if px2 - px1 >= 10 and py2 - py1 >= 10:
                    split_pieces.append(p)

        # 给每块重新分类
        for p in pieces:
            px1, py1, px2, py2 = p
            pw, ph = px2 - px1, py2 - py1
            if pw < 10 or ph < 10:
                continue  # 太小的碎片丢弃
            rtype = _classify_region(px1, py1, px2, py2, img_w, img_h)
            # 如果 reclassify 为 unknown 但父区域有明确类型，继承父类型
            # （避免内容条被切后因 aspect ratio 极端而被误判为 unknown）
            if rtype == "unknown" and r["type"] != "unknown":
                rtype = r["type"]
            refined.append({
                **r,
                "id": 0,
                "bbox": p,
                "type": rtype,
                "desc": rtype,
                "action": _region_action(rtype),
                "split_from": r.get("id"),
            })

    # ─── 第二步：合并相邻同类型小区域 ───
    refined = _merge_adjacent_regions(refined, lines)

    # ─── 第三步：过滤太小的未知碎片（检测噪声） ───
    filtered = []
    for r in refined:
        x1, y1, x2, y2 = r["bbox"]
        bw, bh = x2 - x1, y2 - y1
        if r["type"] == "unknown" and (bw < 25 or bh < 25):
            continue  # 太小且未知 → 检测噪声，丢弃
        filtered.append(r)
    refined = filtered

    # 重编号
    for i, r in enumerate(refined):
        r["id"] = i + 1

    if return_split_vis:
        return refined, split_regions, split_pieces
    return refined


MIN_PIECE_SIZE = 25  # 切分后每块在切分方向上的最小尺寸，避免生成碎片


def _split_piece(bbox, h_lines, v_lines):
    """
    递归切分一个 bbox。

    策略：先水平切，对每一块再垂直切。
    避免生成太小的碎片（< MIN_PIECE_SIZE 的块将放弃切分）。
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
        # 按离中间距离排序
        mid_y = (y1 + y2) // 2
        sorted_h = sorted(my_h, key=lambda l: abs(l["position"] - mid_y))
        for line in sorted_h:
            pos = int(line["position"])
            # 检查两块是否都足够大
            if pos - y1 >= MIN_PIECE_SIZE and y2 - pos >= MIN_PIECE_SIZE:
                top, bottom = _split_horizontal(bbox, pos)
                remaining_h = [l for l in my_h if l != line]
                result = []
                for piece in [top, bottom]:
                    result.extend(_split_piece(piece, remaining_h, my_v))
                return result
        # 没有合适的线能切出足够大的两块 → 放弃切分
        return [bbox]

    # 只有垂直线
    if my_v:
        mid_x = (x1 + x2) // 2
        sorted_v = sorted(my_v, key=lambda l: abs(l["position"] - mid_x))
        for line in sorted_v:
            pos = int(line["position"])
            if pos - x1 >= MIN_PIECE_SIZE and x2 - pos >= MIN_PIECE_SIZE:
                left, right = _split_vertical(bbox, pos)
                remaining_v = [l for l in my_v if l != line]
                result = []
                for piece in [left, right]:
                    result.extend(_split_piece(piece, my_h, remaining_v))
                return result
        return [bbox]

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
