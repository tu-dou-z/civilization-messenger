# -*- coding: utf-8 -*-
"""生成丝路地图底衬影像（NASA Blue Marble 地形阴影 + 水文）

按 mapdata.js 里同一套墨卡托投影从 NASA GIBS 取瓦片，拼成一张 2000×1200 的底图，
并做色阶处理（保留山脉 / 沙漠 / 绿洲层次，压掉卫星图的高饱和蓝绿，
使其能同时被浅色“宣纸”与深色“玄夜”两套主题当底衬用）。

输出: images/map/terrain.jpg
"""
import io
import json
import math
import os

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "images", "map")
OUT_W = 2000
OUT_H = 740        # 与 mapdata.js 的 1000×370 同比例
Z = 6

GIBS = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        "BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg")


def load_bbox():
    raw = open(os.path.join(ROOT, "mapdata.js"), encoding="utf-8").read()
    data = json.loads(raw.split("= ", 1)[1].rstrip().rstrip(";"))
    return data["bbox"]


def lonlat_to_world(lon, lat, z):
    n = 2 ** z * 256
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.0), -85.0)
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def main():
    lon0, lat0, lon1, lat1 = load_bbox()
    x0, y0 = lonlat_to_world(lon0, lat1, Z)   # 左上
    x1, y1 = lonlat_to_world(lon1, lat0, Z)   # 右下
    tx0, ty0 = int(x0 // 256), int(y0 // 256)
    tx1, ty1 = int(x1 // 256), int(y1 // 256)
    w, h = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    mosaic = Image.new("RGB", (w, h), (40, 40, 40))
    print(f"tiles x{tx0}-{tx1} y{ty0}-{ty1} -> {w}x{h}")
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            url = GIBS.format(z=Z, y=ty, x=tx)
            try:
                r = requests.get(url, timeout=40)
                r.raise_for_status()
                tile = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception as e:            # 单块失败不致命，留底色
                print("  ! tile", tx, ty, e)
                continue
            mosaic.paste(tile, (tx * 256 - tx0 * 256, ty * 256 - ty0 * 256))

    crop_box = (int(x0 - tx0 * 256), int(y0 - ty0 * 256),
                int(x1 - tx0 * 256), int(y1 - ty0 * 256))
    img = mosaic.crop(crop_box).resize((OUT_W, OUT_H), Image.LANCZOS)

    a = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = r * 0.299 + g * 0.587 + b * 0.114

    # 水体：GIBS 的阴影地形里海是深蓝，先抠出来（含浅海与湖）
    water_mask = np.clip((b - r) * 4.0 - 0.12, 0, 1)

    # 陆地微起伏：用原图亮度的高通部分当作山体明暗（大尺度去掉，避免整体发灰）
    base = np.asarray(Image.fromarray((lum * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(28))).astype(np.float32) / 255.0
    relief = np.clip((lum - base) * 5.2 + 0.5, 0, 1)      # 0.5 为中性，山脊亮、谷地暗

    # 植被（绿洲带 / 草原 / 河谷）—— GIBS 里绿的地方 g 明显高于 r
    veg = np.clip((g - r) * 5.0 + (g - 0.26) * 2.4, 0, 1) * (1 - water_mask)

    sand = np.array([0.905, 0.815, 0.640])
    ochre = np.array([0.760, 0.620, 0.420])
    jade = np.array([0.475, 0.545, 0.395])

    # 沙色 → 赭色的宏观分布，叠加植被
    tone = np.clip(base * 1.7 - 0.18, 0, 1)[:, :, None]
    land = sand[None, None, :] * (1 - tone) + ochre[None, None, :] * tone
    land = land * (1 - veg[:, :, None] * 0.72) + jade[None, None, :] * (veg[:, :, None] * 0.72)

    # 山体明暗：±22% 的乘性变化，足够看出兴都库什、天山、帕米尔的走向
    shade = 0.74 + relief[:, :, None] * 0.52
    land = land * shade

    water = np.array([0.235, 0.365, 0.430])[None, None, :] * (0.86 + base[:, :, None] * 0.22)
    out = land * (1 - water_mask[:, :, None]) + water * water_mask[:, :, None]

    out = np.clip(out, 0, 1)
    res = Image.fromarray((out * 255).astype(np.uint8))
    res = res.filter(ImageFilter.SMOOTH)
    res = ImageEnhance.Contrast(res).enhance(1.05)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "terrain.jpg")
    res.save(path, quality=86, optimize=True, progressive=True)
    print("written", path, f"{os.path.getsize(path)/1024:.0f} KB")


if __name__ == "__main__":
    main()
