# -*- coding: utf-8 -*-
"""生成丝路真实地理底图数据 (mapdata.js)

数据源:
  - world-atlas countries-50m (Natural Earth 1:50m 国界)
  - ne_50m_lakes / ne_50m_rivers_lake_centerlines (Natural Earth 1:50m)
两者均需先下载到 _geo_src/。

输出: mapdata.js —— 内含投影后的 SVG path 字符串，浏览器端直接 <script src> 加载，
      无需 fetch（file:// 打开也能用）。
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "_geo_src")

# 地图范围（经纬度）：长安 → 大马士革，并向南北留出地理语境
# 画幅 1000×370 是按经度跨度与墨卡托纬度跨度算出的“等比例”高度（1:1，无拉伸），
# 商路横贯画面，上下仍留出中亚草原与波斯湾的语境。
LON0, LON1 = 26.0, 117.0
LAT0, LAT1 = 21.5, 48.5
W, H = 1000.0, 370.0


def merc_y(lat):
    lat = max(min(lat, 85.0), -85.0)
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def merc_lat(y):
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)


Y0, Y1 = merc_y(LAT1), merc_y(LAT0)   # 上边界、下边界（mercator y 越大越靠北）
X0, X1 = math.radians(LON0), math.radians(LON1)


def project(lon, lat):
    x = (math.radians(lon) - X0) / (X1 - X0) * W
    y = (Y0 - merc_y(lat)) / (Y0 - Y1) * H
    return x, y


# ---------------------------------------------------------------- topojson
def decode_arcs(topo):
    tr = topo.get("transform")
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        out = []
        for dx, dy in arc:
            x += dx
            y += dy
            if tr:
                out.append((x * tr["scale"][0] + tr["translate"][0],
                            y * tr["scale"][1] + tr["translate"][1]))
            else:
                out.append((x, y))
        arcs.append(out)
    return arcs


def ring_from_arcs(arc_idx, arcs):
    pts = []
    for i in arc_idx:
        a = arcs[i] if i >= 0 else list(reversed(arcs[~i]))
        pts.extend(a if not pts else a[1:])
    return pts


def polygons_of(geom, arcs):
    t = geom["type"]
    if t == "Polygon":
        return [[ring_from_arcs(r, arcs) for r in geom["arcs"]]]
    if t == "MultiPolygon":
        return [[ring_from_arcs(r, arcs) for r in poly] for poly in geom["arcs"]]
    return []


# ---------------------------------------------------------------- 几何工具
def bbox_of(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _signed_area(ring):
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return s / 2.0


def ring_area(ring):
    return abs(_signed_area(ring))


def intersects(bb, pad=1.0):
    return not (bb[2] < LON0 - pad or bb[0] > LON1 + pad or
                bb[3] < LAT0 - pad or bb[1] > LAT1 + pad)


def clip_ring(ring, b=(LON0, LAT0, LON1, LAT1)):
    """Sutherland–Hodgman：用矩形裁剪环。"""
    def clip_edge(pts, inside, inter):
        if not pts:
            return []
        out = []
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
            prev = cur
        return out

    def ix_x(v, p, q):
        k = (v - p[0]) / (q[0] - p[0]) if q[0] != p[0] else 0
        return (v, p[1] + (q[1] - p[1]) * k)

    def ix_y(v, p, q):
        k = (v - p[1]) / (q[1] - p[1]) if q[1] != p[1] else 0
        return (p[0] + (q[0] - p[0]) * k, v)

    xmin, ymin, xmax, ymax = b
    pts = ring
    pts = clip_edge(pts, lambda p: p[0] >= xmin, lambda p, q: ix_x(xmin, p, q))
    pts = clip_edge(pts, lambda p: p[0] <= xmax, lambda p, q: ix_x(xmax, p, q))
    pts = clip_edge(pts, lambda p: p[1] >= ymin, lambda p, q: ix_y(ymin, p, q))
    pts = clip_edge(pts, lambda p: p[1] <= ymax, lambda p, q: ix_y(ymax, p, q))
    return pts


def simplify(pts, tol):
    """Douglas–Peucker（闭合环：先断开首尾重复点）。"""
    if len(pts) < 4:
        return pts
    closed = pts[0] == pts[-1]
    ring = pts[:-1] if closed else pts

    def dp(seq):
        if len(seq) < 3:
            return seq
        (x1, y1), (x2, y2) = seq[0], seq[-1]
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy)
        best, bi = -1, 0
        for i in range(1, len(seq) - 1):
            px, py = seq[i]
            d = abs(dy * (px - x1) - dx * (py - y1)) / n if n else math.hypot(px - x1, py - y1)
            if d > best:
                best, bi = d, i
        if best <= tol:
            return [seq[0], seq[-1]]
        left = dp(seq[:bi + 1])
        right = dp(seq[bi:])
        return left[:-1] + right

    # 闭合环取最远点切分，避免首尾重合导致退化
    far = max(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - ring[0][0],
                                                        ring[i][1] - ring[0][1]))
    a = dp(ring[:far + 1])
    b = dp(ring[far:] + [ring[0]])
    out = a[:-1] + b[:-1]
    return out + [out[0]] if out else out


def to_path(rings, tol=0.35, precision=1):
    parts = []
    for r in rings:
        r = simplify(r, tol)
        if len(r) < 4:
            continue
        seg = []
        for i, (lon, lat) in enumerate(r):
            x, y = project(lon, lat)
            seg.append(("M" if i == 0 else "L") + f"{x:.{precision}f} {y:.{precision}f}")
        parts.append("".join(seg) + "Z")
    return "".join(parts)


# ---------------------------------------------------------------- 国家名
NAME_ZH = {
    "China": "中国", "Kazakhstan": "哈萨克斯坦", "Uzbekistan": "乌兹别克斯坦",
    "Turkmenistan": "土库曼斯坦", "Tajikistan": "塔吉克斯坦", "Kyrgyzstan": "吉尔吉斯斯坦",
    "Iran": "伊朗", "Iraq": "伊拉克", "Syria": "叙利亚", "Turkey": "土耳其",
    "Egypt": "埃及", "India": "印度", "Pakistan": "巴基斯坦", "Afghanistan": "阿富汗",
    "Russia": "俄罗斯", "Saudi Arabia": "沙特阿拉伯", "Jordan": "约旦", "Lebanon": "黎巴嫩",
    "Israel": "以色列", "Azerbaijan": "阿塞拜疆", "Armenia": "亚美尼亚", "Georgia": "格鲁吉亚",
    "Nepal": "尼泊尔", "Mongolia": "蒙古", "Oman": "阿曼",
    "United Arab Emirates": "阿联酋", "Qatar": "卡塔尔", "Kuwait": "科威特",
    "Libya": "利比亚", "Sudan": "苏丹", "Ethiopia": "埃塞俄比亚", "Somalia": "索马里",
    "Yemen": "也门", "Greece": "希腊", "Bulgaria": "保加利亚", "Ukraine": "乌克兰",
    "Romania": "罗马尼亚", "Italy": "意大利", "Cyprus": "塞浦路斯", "Bangladesh": "孟加拉国",
    "Myanmar": "缅甸", "Thailand": "泰国", "Vietnam": "越南", "Laos": "老挝",
    "Bhutan": "不丹", "Sri Lanka": "斯里兰卡", "Maldives": "马尔代夫", "Palestine": "巴勒斯坦",
    "North Macedonia": "北马其顿", "Serbia": "塞尔维亚", "Albania": "阿尔巴尼亚",
    "North Korea": "朝鲜", "South Korea": "韩国", "Japan": "日本", "Moldova": "摩尔多瓦",
    "Hungary": "匈牙利", "Slovakia": "斯洛伐克", "Poland": "波兰", "Belarus": "白俄罗斯",
    "Eritrea": "厄立特里亚", "Djibouti": "吉布提", "Kenya": "肯尼亚",
    "South Sudan": "南苏丹", "Chad": "乍得", "Niger": "尼日尔", "Algeria": "阿尔及利亚",
    "Tunisia": "突尼斯", "Mali": "马里", "Mauritania": "毛里塔尼亚", "Senegal": "塞内加尔",
    "Cameroon": "喀麦隆", "Nigeria": "尼日利亚", "Central African Republic": "中非共和国",
    "Republic of the Congo": "刚果（布）", "Democratic Republic of the Congo": "刚果（金）",
    "Uganda": "乌干达", "Tanzania": "坦桑尼亚", "Zambia": "赞比亚", "Angola": "安哥拉",
    "Namibia": "纳米比亚", "Botswana": "博茨瓦纳", "Zimbabwe": "津巴布韦",
    "Mozambique": "莫桑比克", "Malawi": "马拉维", "Madagascar": "马达加斯加",
    "Burkina Faso": "布基纳法索", "Ghana": "加纳", "Côte d'Ivoire": "科特迪瓦",
    "Guinea": "几内亚", "Sierra Leone": "塞拉利昂", "Liberia": "利比里亚",
    "Togo": "多哥", "Benin": "贝宁", "Gabon": "加蓬", "Equatorial Guinea": "赤道几内亚",
    "Gambia": "冈比亚", "Guinea-Bissau": "几内亚比绍", "Morocco": "摩洛哥",
    "Western Sahara": "西撒哈拉", "Kosovo": "科索沃", "Montenegro": "黑山",
    "Bosnia and Herz.": "波黑", "Croatia": "克罗地亚", "Slovenia": "斯洛文尼亚",
    "Austria": "奥地利", "Czechia": "捷克", "Germany": "德国", "Lithuania": "立陶宛",
    "Latvia": "拉脱维亚", "Estonia": "爱沙尼亚", "Finland": "芬兰", "Sweden": "瑞典",
    "Norway": "挪威", "Denmark": "丹麦", "Netherlands": "荷兰", "Belgium": "比利时",
    "United Kingdom": "英国", "Ireland": "爱尔兰", "France": "法国", "Spain": "西班牙",
    "Portugal": "葡萄牙", "Switzerland": "瑞士", "Malta": "马耳他",
    "United States of America": "美国", "Canada": "加拿大", "Mexico": "墨西哥",
    "Indonesia": "印度尼西亚", "Malaysia": "马来西亚", "Philippines": "菲律宾",
    "Brunei": "文莱", "Timor-Leste": "东帝汶", "Singapore": "新加坡",
}

NAME_AR = {
    "China": "الصين", "Kazakhstan": "كازاخستان", "Uzbekistan": "أوزبكستان",
    "Turkmenistan": "تركمانستان", "Tajikistan": "طاجيكستان", "Kyrgyzstan": "قيرغيزستان",
    "Iran": "إيران", "Iraq": "العراق", "Syria": "سوريا", "Turkey": "تركيا",
    "Egypt": "مصر", "India": "الهند", "Pakistan": "باكستان", "Afghanistan": "أفغانستان",
    "Russia": "روسيا", "Saudi Arabia": "السعودية", "Jordan": "الأردن", "Lebanon": "لبنان",
    "Israel": "إسرائيل", "Azerbaijan": "أذربيجان", "Armenia": "أرمينيا", "Georgia": "جورجيا",
    "Nepal": "نيبال", "Mongolia": "منغوليا", "Oman": "عُمان", "United Arab Emirates": "الإمارات",
    "Qatar": "قطر", "Kuwait": "الكويت", "Libya": "ليبيا", "Sudan": "السودان",
    "Ethiopia": "إثيوبيا", "Somalia": "الصومال", "Yemen": "اليمن", "Greece": "اليونان",
    "Bulgaria": "بلغاريا", "Ukraine": "أوكرانيا", "Romania": "رومانيا", "Italy": "إيطاليا",
    "Cyprus": "قبرص", "Bangladesh": "بنغلاديش", "Myanmar": "ميانمار", "Thailand": "تايلاند",
    "Vietnam": "فيتنام", "Laos": "لاوس", "Bhutan": "بوتان", "Sri Lanka": "سريلانكا",
    "Maldives": "المالديف", "Palestine": "فلسطين", "North Korea": "كوريا الشمالية",
    "South Korea": "كوريا الجنوبية", "Japan": "اليابان",
}

# 画面内不标注的名字：碎片化小国，以及 Natural Earth 的区域标注（冰川区等）
LABEL_SKIP = {"Israel", "Lebanon", "Qatar", "Kuwait", "Cyprus", "N. Cyprus", "Palestine",
              "Djibouti", "Bahrain", "Singapore", "Brunei", "Malta", "Kosovo",
              "Montenegro", "Armenia", "Georgia", "Azerbaijan", "Maldives",
              "Bhutan", "Sri Lanka", "Timor-Leste", "Gambia", "Guinea-Bissau",
              "Equatorial Guinea", "North Macedonia", "Albania", "Bosnia and Herz.",
              "Moldova", "Jordan", "United Arab Emirates", "Oman",
              "Eritrea", "Bangladesh", "Nepal", "Tajikistan", "Kyrgyzstan",
              "Siachen Glacier", "Macao", "Hong Kong", "Macedonia"}


def main():
    topo = json.load(open(os.path.join(SRC, "_world50.json"), encoding="utf-8"))
    arcs = decode_arcs(topo)
    geoms = topo["objects"]["countries"]["geometries"]

    countries = []
    for g in geoms:
        name = (g.get("properties") or {}).get("name") or ""
        polys = polygons_of(g, arcs)
        if not polys:
            continue
        flat = [r for poly in polys for r in poly]
        bb = (min(bbox_of(r)[0] for r in flat), min(bbox_of(r)[1] for r in flat),
              max(bbox_of(r)[2] for r in flat), max(bbox_of(r)[3] for r in flat))
        if not intersects(bb):
            continue
        rings = []
        for poly in polys:
            for ri, ring in enumerate(poly):
                if len(ring) < 3:
                    continue
                clipped = clip_ring(ring)
                if len(clipped) < 3:
                    continue
                # 面积过小的碎块直接丢弃（洞里的小岛、投影裁切残片）
                xs = [p[0] for p in clipped]
                ys = [p[1] for p in clipped]
                if (max(xs) - min(xs)) < 0.05 and (max(ys) - min(ys)) < 0.05:
                    continue
                rings.append(clipped)
        if not rings:
            continue
        d = to_path(rings, tol=0.02)
        if not d:
            continue
        # 注记锚点：取投影面积最大的一块（避免小岛把名字拉偏），用其包围盒中心
        biggest = max(rings, key=ring_area)
        bx0, by0, bx1, by1 = bbox_of(biggest)
        lx, ly = (bx0 + bx1) / 2, (by0 + by1) / 2
        cx, cy = project(lx, ly)
        countries.append({
            "id": name, "zh": NAME_ZH.get(name, name), "ar": NAME_AR.get(name, name),
            "d": d,
            "label": name not in LABEL_SKIP,
            "lx": round(cx, 1), "ly": round(cy, 1),
            "ac": round(biggest and abs(_signed_area(biggest)), 1),
        })

    # 湖
    lakes = []
    lk = json.load(open(os.path.join(SRC, "_ne_50m_lakes.geojson"), encoding="utf-8"))
    for ft in lk["features"]:
        p = ft["properties"]
        geom = ft["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        rings = []
        for poly in polys:
            for ring in poly:
                if len(ring) < 3:
                    continue
                bb = bbox_of(ring)
                if not intersects(bb):
                    continue
                c = clip_ring(ring)
                if len(c) >= 3:
                    rings.append(c)
        if rings:
            lakes.append({"name": p.get("name") or "", "d": to_path(rings, tol=0.01)})

    # 河
    rivers = []
    rv = json.load(open(os.path.join(SRC, "_ne_50m_rivers_lake_centerlines.geojson"),
                        encoding="utf-8"))
    for ft in rv["features"]:
        p = ft["properties"]
        if (p.get("scalerank") or 99) > 6:
            continue
        geom = ft["geometry"]
        lines = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        parts = []
        for line in lines:
            bb = bbox_of(line)
            if not intersects(bb):
                continue
            seg = []
            for i, (lon, lat) in enumerate(simplify(line, 0.03)):
                x, y = project(lon, lat)
                seg.append(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}")
            if len(seg) > 1:
                parts.append("".join(seg))
        if parts:
            rivers.append({"name": p.get("name") or "", "d": "".join(parts)})

    # 陆地整体轮廓（用于把“海色/陆色”只刷在正确的区域上）
    land_rings = []
    land_topo = json.load(open(os.path.join(SRC, "_land50.json"), encoding="utf-8"))
    land_arcs = decode_arcs(land_topo)
    for g in land_topo["objects"]["land"]["geometries"]:
        for poly in polygons_of(g, land_arcs):
            for ring in poly:
                if len(ring) < 3:
                    continue
                if not intersects(bbox_of(ring), pad=4.0):
                    continue
                c = clip_ring(ring)
                if len(c) >= 3:
                    land_rings.append(c)
    land_d = to_path(land_rings, tol=0.03)

    out = {
        "w": W, "h": H, "bbox": [LON0, LAT0, LON1, LAT1],
        "land": land_d, "countries": countries, "lakes": lakes, "rivers": rivers,
    }
    js = ("/* 丝路真实地理底图数据 · 自动生成（Natural Earth 1:50m），请勿手工编辑 */\n"
          "window.CIVMAP = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    path = os.path.join(ROOT, "mapdata.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"bbox lon {LON0}~{LON1:.0f}, lat {LAT0:.2f}~{LAT1:.2f}")
    print(f"countries {len(countries)}  lakes {len(lakes)}  rivers {len(rivers)}")
    print(f"written {path}  ({len(js)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
