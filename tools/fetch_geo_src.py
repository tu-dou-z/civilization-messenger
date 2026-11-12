# -*- coding: utf-8 -*-
"""下载 build_map.py 所需的 Natural Earth 1:50m 源数据到 _geo_src/

用法：
    python tools/fetch_geo_src.py

下载完成后即可运行：
    python tools/build_map.py        # 重新生成 mapdata.js

说明：
  这四个文件是 build_map.py 的唯一输入，共约 2.9 MB。
  已存在且大小正确的文件会跳过，不重复下载。

数据来源：
  - world-atlas@2 (TopoJSON, 由 Natural Earth 1:50m 转换)
      countries-50m.json / land-50m.json
  - nvkelso/natural-earth-vector (官方 GeoJSON 镜像)
      ne_50m_lakes / ne_50m_rivers_lake_centerlines
"""
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("缺少 requests：请先运行 pip install requests")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "_geo_src")

# (文件名, 下载地址, 预期字节数)  —— 预期大小用于校验下载是否完整
SOURCES = [
    ("_world50.json",
     "https://unpkg.com/world-atlas@2/countries-50m.json", 756420),
    ("_land50.json",
     "https://unpkg.com/world-atlas@2/land-50m.json", 545534),
    ("_ne_50m_lakes.geojson",
     "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",
     876018),
    ("_ne_50m_rivers_lake_centerlines.geojson",
     "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_rivers_lake_centerlines.geojson",
     827318),
]

# 上面的 raw.githubusercontent.com 在国内常被墙，逐个回退到可用的镜像
MIRRORS = [
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/",
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/",
    "https://ghproxy.net/https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/",
    "https://raw.gitmirror.com/nvkelso/natural-earth-vector/master/geojson/",
]


def _alt_urls(url):
    """给 Natural Earth 的地址生成镜像候选列表。"""
    name = url.rsplit("/", 1)[-1]
    if "nvkelso/natural-earth-vector" not in url:
        return [url]
    return [m + name for m in MIRRORS]


def fetch(name, url, expect):
    path = os.path.join(SRC, name)
    if os.path.exists(path) and os.path.getsize(path) == expect:
        print(f"跳过 {name}（已存在，{expect} 字节）")
        return True

    for alt in _alt_urls(url):
        try:
            print(f"下载 {name} ... {alt}")
            r = requests.get(alt, timeout=90)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}，换下一个源")
                continue
            if len(r.content) != expect:
                print(f"  大小不符（{len(r.content)} != {expect}），换下一个源")
                continue
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"  完成 {len(r.content)} 字节")
            return True
        except requests.RequestException as e:
            print(f"  失败：{e}")
            continue

    print(f"!! {name} 全部数据源均失败")
    return False


def main():
    os.makedirs(SRC, exist_ok=True)
    print(f"目标目录：{SRC}\n")

    ok = True
    for name, url, expect in SOURCES:
        ok = fetch(name, url, expect) and ok

    print()
    if ok:
        print("全部就绪。下一步运行： python tools/build_map.py")
    else:
        print("有文件下载失败，请检查网络后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
