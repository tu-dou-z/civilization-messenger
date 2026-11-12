"""文明信使 · 本地后端代理
把前端 /api/chat 请求转发到 OfoxAI，API Key 只保存在后端。
运行：python app.py   然后浏览器打开 index.html
"""
import ast
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, jsonify, request, send_from_directory

OFOX_URL = "https://api.ofox.io/v1/chat/completions"
OFOX_MODEL = os.environ.get("OFOX_MODEL", "deepseek/deepseek-v4.1-flash-1017")

# 「文明信使」人格提示词：后端强制注入，保证每次对话都有稳定的语气与知识边界
CHAT_SYSTEM_PROMPT = """你是一个文化历史专家。请严格按照以下 JSON 格式回答：
{
  "reply": "用100字以内，准确、生动地讲述用户提问的【具体物品】的故事或特点。绝不允许泛泛而谈国家历史，必须紧扣该物品。",
  "search_keyword": "提取用户提问中的【核心实体物品名词】。比如问‘阿拉伯飞毯的故事’，提取‘飞毯’；问‘大马士革铜器’，提取‘大马士革铜器’。绝对不要保留‘故事’、‘阿拉伯’等修饰词。"
}"""
# 图片识别专用模型（视觉 + 极低消耗）。
# 可填多个，逗号分隔，按顺序尝试——某个模型被上游下线时自动回退到下一个。
# 注意：gpt-4o-mini 已于上游下线（HTTP 410 model_deprecated）。
# 不要用带 reasoning 的模型（如 gpt-5-nano / qwen3.5-flash），
# 推理 token 会把 max_tokens 吃光，导致 content 为空。
OFOX_VISION_MODELS = [
    m.strip()
    for m in os.environ.get(
        "OFOX_VISION_MODEL",
        "google/gemini-2.5-flash-lite,z-ai/glm-5v-turbo,google/gemini-2.5-flash",
    ).split(",")
    if m.strip()
]
# API Key 从环境变量 OFOX_API_KEY 读取，不写在代码里
OFOX_KEY = os.environ.get("OFOX_API_KEY", "").strip()

RECOGNIZE_SYSTEM_PROMPT = """你是一个中阿文化交流专家。请识别图片中的主体物品（可以是任何文化符号、工艺品、建筑、服饰等）。请严格返回如下 JSON 格式：
{"name_zh": "中文名称（如：宫灯）", "name_ar": "阿拉伯语名称", "desc_zh": "一段非常精简的中文介绍（严格控制在50字以内）", "desc_ar": "对应的阿拉伯语精简介绍", "theme": "从[cn, ar, ca, in, eg]中选一个最匹配的产地/风格代号"}
如果你实在认不出图片里是什么，才返回 {"error": "unknown"}。不要输出除 JSON 外的任何废话。"""

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))

# 允许通过 HTTP 访问的静态资源后缀（不含 .py / .bat / .zip / .doc 等）
STATIC_EXTS = {
    ".html", ".css", ".js", ".json", ".txt",
    ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".ico",
    ".woff", ".woff2", ".ttf",
}

# 后缀合法但仍属内部文件，单独按文件名拦掉
DENY_FILES = {
    "requirements.txt", "render.yaml", "部署说明.md",
    "readme.md", "readme.txt", "license", "license.txt",
}


@app.route("/")
def index():
    """浏览器访问 http://127.0.0.1:5000 即可打开站点（也便于公网部署）。"""
    return send_from_directory(ROOT, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """托管 index.html 引用的同级静态资源（css / js / 图片 / 素材）。

    index.html 与前端资源都在项目根目录，不在 Flask 默认的 static/ 下，
    因此这里显式托管。只放行白名单后缀，避免 app.py、.bat、.zip、
    _geo_src 等源码与配置被下载。
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in STATIC_EXTS:
        return jsonify({"error": "not_found"}), 404
    # 以 _ 开头的目录/文件视为内部资源（_geo_src、_分享包、_前缀备份等），不对外
    parts = filename.replace("\\", "/").split("/")
    if any(p.startswith("_") or p == "tools" for p in parts):
        return jsonify({"error": "not_found"}), 404
    if parts[-1].lower() in DENY_FILES:
        return jsonify({"error": "not_found"}), 404
    full = os.path.abspath(os.path.join(ROOT, filename))
    if not full.startswith(ROOT + os.sep) or not os.path.isfile(full):
        return jsonify({"error": "not_found"}), 404
    return send_from_directory(ROOT, filename)


def _parse_json_loose(text):
    """尽量从模型输出里取出 JSON（容忍 ``` 包裹、前后废话、单引号）。"""
    if not isinstance(text, str):
        return None
    t = text.strip().strip("﻿​").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    candidates = [t]
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            v = json.loads(c)
            return v if isinstance(v, dict) else None
        except ValueError:
            pass
        try:  # 模型偶尔照抄提示词里的单引号写法
            v = ast.literal_eval(c)
            return v if isinstance(v, dict) else None
        except (ValueError, SyntaxError):
            pass
    return None


@app.after_request
def add_cors_headers(resp):
    # 本地 file:// 打开（Origin: null）与本站同源页面需要跨域头；
    # 不再无条件放开 "*"，配合 _same_origin() 把第三方站点挡在外面。
    origin = request.headers.get("Origin")
    if origin == "null":
        resp.headers["Access-Control-Allow-Origin"] = "null"
    elif origin and _same_origin():
        resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def _same_origin():
    """只接受本站页面发起的 AI 请求。

    挡住「别人把自己的网页挂上去、用访问者的浏览器白白消耗我们额度」这一路。
    放行三种情况：没有 Origin（curl/服务端调用）、Origin 为 null（file:// 本地打开）、
    Origin 主机名与本站一致。注意这是降低滥用面，不是密码学级别的鉴权。
    """
    origin = request.headers.get("Origin")
    if not origin or origin == "null":
        return True
    try:
        host = origin.split("//", 1)[1].split("/")[0]
    except IndexError:
        return False
    req_host = request.headers.get("Host", "")
    return host == req_host


# ---------------------------------------------------------------- 频率限制
# 公网部署时防止单个 IP 刷爆 AI 额度。内存计数，进程重启后清零；
# 多实例部署需换成 Redis 等共享存储。
RATE_LIMITS = {
    "chat":      (20, 60),    # 每 60 秒最多 20 次
    "recognize": (10, 60),    # 每 60 秒最多 10 次
    "bilibili":  (15, 60),    # 每 60 秒最多 15 次
}
_rate_bucket = {}             # {(ip, name): [时间戳, ...]}
_rate_lock = threading.Lock()


# ---------------------------------------------------------------- 全局日额度
# 兜底防线：即使有人拿到链接、换 IP 轮番请求绕过了单 IP 限流，
# 一天的 AI 调用总量也有硬上限，被盗刷的损失封顶在可接受范围。
# 计数在内存里，进程重启（含免费实例休眠）会清零，
# 所以它是「防大额盗刷」而不是「精确计费」。
DAILY_QUOTA = {
    "chat":      int(os.environ.get("QUOTA_CHAT", "600")),
    "recognize": int(os.environ.get("QUOTA_RECOGNIZE", "400")),
    "bilibili":  int(os.environ.get("QUOTA_BILIBILI", "400")),
}
_daily = {}                   # {name: 今日次数}
_daily_day = [""]             # 当前统计日期，换天即清零
_daily_lock = threading.Lock()


def over_quota(name):
    """今日该接口调用量是否已达上限。"""
    limit = DAILY_QUOTA.get(name)
    if not limit:
        return False
    day = time.strftime("%Y-%m-%d")
    with _daily_lock:
        if _daily_day[0] != day:      # 跨天自动重置，字典不会无限增长
            _daily.clear()
            _daily_day[0] = day
        if _daily.get(name, 0) >= limit:
            return True
        _daily[name] = _daily.get(name, 0) + 1
    return False


def _client_ip():
    # 云平台通常经反向代理，优先取 X-Forwarded-For 的第一个地址
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() or request.remote_addr or "unknown")


def rate_limited(name):
    """命中限流返回 True。"""
    limit = RATE_LIMITS.get(name)
    if not limit:
        return False
    times, window = limit
    key = (_client_ip(), name)
    now = time.time()
    with _rate_lock:
        stamps = [t for t in _rate_bucket.get(key, []) if now - t < window]
        if len(stamps) >= times:
            _rate_bucket[key] = stamps
            return True
        stamps.append(now)
        _rate_bucket[key] = stamps
    return False


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


def _guard(name):
    """AI 接口统一前置检查：来源校验 → 单 IP 限流 → 全局日额度。
    返回 (响应, 状态码) 表示应直接返回；返回 None 表示放行。"""
    if request.method == "OPTIONS":
        return "", 204
    if not _same_origin():
        return jsonify({"error": "forbidden"}), 403
    if rate_limited(name):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429
    if over_quota(name):
        return jsonify({"error": "今日演示额度已用完，请明天再试"}), 429
    return None


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    blocked = _guard("chat")
    if blocked:
        return blocked

    if not OFOX_KEY:
        return jsonify({"error": "后端未设置环境变量 OFOX_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages 不能为空"}), 400

    # 强制在 messages 最前面加入「文明信使」人格提示词（忽略前端是否已带 system）
    messages = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    messages.insert(0, {"role": "system", "content": CHAT_SYSTEM_PROMPT})

    def _post(body):
        return requests.post(
            OFOX_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OFOX_KEY}",
            },
            json=body,
            timeout=60,
        )

    body = {
        "model": data.get("model") or OFOX_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    try:
        r = _post(body)
        if r.status_code == 400 and "response_format" in r.text:
            body.pop("response_format", None)  # 网关不支持 json_object 时去掉重试
            r = _post(body)
    except requests.Timeout:
        return jsonify({"error": "请求 OfoxAI 超时"}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"无法连接 OfoxAI：{e}"}), 502

    try:
        payload = r.json()
    except ValueError:
        payload = {"error": r.text[:500]}
    if not r.ok:
        print(f"[OfoxAI] HTTP {r.status_code}: {str(payload)[:500]}")
        return jsonify(payload), r.status_code

    # 后端把模型输出的 JSON 字符串解析好，前端直接用 reply + search_keyword
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""
    parsed = _parse_json_loose(content) or {}
    reply = str(parsed.get("reply") or content or "").strip()
    # 系统提示词要求 100 字以内的 reply，这里再截一刀，防止模型跑题写成长文
    reply = reply[:400]
    keyword = str(parsed.get("search_keyword") or "").strip().strip("「」《》\"'")
    # 去掉“的/的故事/历史”这类修饰，保留实体本身（含产地，如“大马士革铜器”）
    keyword = re.sub(r"的(故事|历史|来源|起源|特点|介绍)$", "", keyword).strip()[:20]
    if not keyword:
        # 兜底：从用户最后一句话里截一个实体词，绝不让搜索落空
        last_user = next(
            (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        keyword = re.sub(
            r"的?(故事|历史|来源|起源|特点|介绍|是什么|怎么样|吗|呢|？|\?|。|，|,)$",
            "",
            str(last_user),
        ).strip()[:20]
    return jsonify({"reply": reply, "search_keyword": keyword})


@app.route("/api/recognize", methods=["POST", "OPTIONS"])
def recognize():
    """接收 Base64 图片 → 视觉模型识别 → 原样返回 JSON。"""
    blocked = _guard("recognize")
    if blocked:
        return blocked

    if not OFOX_KEY:
        return jsonify({"error": "recognition_failed", "detail": "未设置 OFOX_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    image = data.get("image")
    if not isinstance(image, str) or not image.strip():
        return jsonify({"error": "recognition_failed", "detail": "缺少 image 字段"}), 400
    if not image.startswith("data:"):  # 兼容纯 Base64
        image = "data:image/jpeg;base64," + image.strip()

    messages = [
        {"role": "system", "content": RECOGNIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "识别这张图片。"},
                # detail=low：单张图仅 85 tokens，成本极低
                {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
            ],
        },
    ]

    def _post(model, with_json_mode):
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0,
        }
        if with_json_mode:
            body["response_format"] = {"type": "json_object"}
        return requests.post(
            OFOX_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OFOX_KEY}",
            },
            json=body,
            timeout=30,
        )

    # 依次尝试候选模型；上游 4xx（模型下线、不支持该参数等）就换下一个
    result = None
    last_status = None
    for model in OFOX_VISION_MODELS:
        try:
            r = _post(model, True)
            if r.status_code == 400 and "response_format" in r.text:
                # 部分网关不支持 json_object，去掉后重试
                r = _post(model, False)
        except requests.Timeout:
            print(f"[recognize] {model} 超时，尝试下一个")
            continue
        except requests.RequestException as e:
            print(f"[recognize] {model} 连接失败：{e}")
            continue

        last_status = r.status_code
        if not r.ok:
            print(f"[recognize] {model} HTTP {r.status_code}: {r.text[:300]}")
            # 4xx 多为“模型不可用/参数不支持”，换下一个；5xx 也继续试
            continue

        try:
            payload = r.json()
        except ValueError:
            print(f"[recognize] {model} 返回非 JSON：{r.text[:300]}")
            continue

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None

        parsed = _parse_json_loose(content)
        if not parsed:
            print(f"[recognize] {model} 无法解析模型输出：{str(content)[:300]}")
            continue
        if parsed.get("name_zh"):
            result = parsed
            break
        if parsed.get("error"):
            # 模型明确表示“认不出”，不必再换模型了
            return jsonify({"error": "unknown"})

    if not result:
        # 全部模型都失败：是服务故障，不是“认不出”，别伪装成 unknown
        print(f"[recognize] 所有视觉模型均失败，最后一次 HTTP {last_status}")
        return jsonify({"error": "recognition_failed", "detail": "视觉模型均不可用"}), 502

    return jsonify(result)


BILI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_bili_session = None
_bili_retry = False   # 4 路并发请求共用一个 Session，重试交给调用方统一做


def _get_bili_session():
    """复用 session：先访问首页拿到 buvid3 等 cookie，否则搜索接口会偶发反爬。"""
    global _bili_session
    if _bili_session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": BILI_UA, "Referer": "https://www.bilibili.com/"})
        try:
            s.get("https://www.bilibili.com/", timeout=8)
        except requests.RequestException:
            pass  # 拿不到 cookie 也照样试
        _bili_session = s
    return _bili_session


# 明显与“文化物品”无关的分区 / 标题词：命中即丢弃，宁可没视频也不能推错
_BILI_BAD_TYPE = ("游戏", "手游", "音游", "动画综合", "娱乐", "数码", "科技", "鬼畜", "翻唱")
_BILI_BAD_TITLE = (
    "攻略", "坐骑", "皮肤", "抽卡", "开箱", "直播", "录屏", "asmr", "鬼畜",
    "上分", "阵容", "版本", "mod", "我的世界", "泰拉瑞亚", "ff14",
    "第五人格", "最强蜗牛", "无损", "音质", "纯音乐", "翻唱", "钢琴",
    "mv", "dj", "hi-res", "1.2x", "0.8x", "音乐", "歌曲",
    "周杰伦", "演唱", "舞台", "舞蹈", "单曲", "循环", "战袍", "歌",
    "外星", "战士", "战机", "宇宙", "磁铁", "段子", "整活", "ai生成",
    # 卡通/玩梗/带货（"班长的魔法飞毯""曹氏阿拉丁神灯"这类标题字面像、内容无关）
    "班长", "曹氏", "原视频", "沙雕", "搞笑", "测评", "推币", "盲盒",
    "skylab", "演示", "发射", "卫星", "军演", "军工", "导弹", "主战坦克",
    "song",
)

# 产地别名：B站标题常用国家/大区名，而不是具体城市名
_BILI_PLACE_ALIAS = {
    "大马士革": ["叙利亚"],
    "巴格达": ["伊拉克"],
    "撒马尔罕": ["乌兹别克斯坦", "中亚"],
    "景德镇": ["中国"],
    "敦煌": [],
    "开罗": ["埃及"],
    "迪拜": ["阿联酋"],
}


# 物品词 → B站里这件事物常用的叫法（末两字并不总是类名）
_BILI_ITEM_WORDS = {
    "飞毯": "飞毯",
    "大马士革铜器": "铜器",
    "大马士革铜": "铜器",
    "阿拉丁神灯": "神灯",
    "青花瓷": "瓷",
    "丝绸": "丝绸",
    "敦煌壁画": "壁画",
    "椰枣": "椰枣",
    "阿拉伯书法": "书法",
    "藏红花": "藏红花",
    "纸莎草画": "纸莎草",
}

# 同义词 / 上位词：B站里这类物品常以别的名字出现（飞毯 → 阿拉丁、天方夜谭）
_BILI_ITEM_ALIAS = {
    "飞毯": ["阿拉丁 飞毯", "天方夜谭 飞毯", "一千零一夜 飞毯", "阿拉伯 飞毯"],
    "阿拉丁神灯": ["一千零一夜 神灯"],
    "青花瓷": ["景德镇 青花瓷 历史", "青花瓷 瓷器 历史"],
}

# 分区黑名单：同名歌曲 / 游戏道具最容易混进来
_BILI_BAD_TYPE_STRICT = (
    "游戏", "手游", "音游", "桌游", "综艺", "音乐", "演奏", "MV", "舞蹈",
    "翻唱", "乐评", "电子竞技", "动画", "电台", "鬼畜", "VOCALOID", "音乐现场",
    "校园学习", "影视剪辑", "亲子", "预告",
)

# 标题里出现这些词，说明讲的是别的东西（同名词条 / 数码产品 / 娱乐）
_BILI_BAD_CONTEXT = (
    "芯片", "半导体", "键盘", "鼠标", "耳机", "手机", "球鞋", "穿搭",
    "美甲", "口红", "护肤", "健身", "戏腔", "材质", "教程", "接龙", "拼花",
    "搭建", "积木", "手工课",
)


def _bili_seconds(dur):
    """B站的 duration 形如 '1:52' 或 '97:55'，转成秒。"""
    parts = str(dur or "").split(":")
    try:
        nums = [int(x) for x in parts]
    except ValueError:
        return 0
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def _bili_search(keyword):
    """搜一次 B站视频。被限流（返回空/非 JSON）时冷却重试一次。"""
    for attempt in range(2 if _bili_retry else 1):
        try:
            r = _get_bili_session().get(
                "https://api.bilibili.com/x/web-interface/search/type",
                params={"search_type": "video", "keyword": keyword, "order": "totalrank"},
                timeout=6,
            )
            payload = r.json()
        except Exception as e:
            print(f"[search_bilibili] 检索 {keyword} 失败：{e}")
            payload = {}
        if payload.get("code") == 0:
            return (payload.get("data") or {}).get("result") or []
        if attempt == 0:
            time.sleep(0.6)
    return []


def _bili_candidates(keyword, limit=6):
    """筛出「标题里确实出现这件物品」且不是游戏/音乐/玩梗的候选。

    查询按优先级排队（原词 → 产地补搜 → 别名补搜），凑够 limit 条就停，
    避免一次问答打七八个请求把页面拖慢。
    """
    core = re.sub(r"\s+", "", keyword)
    item = _BILI_ITEM_WORDS.get(core) or (core[-2:] if len(core) > 2 else core)
    place = core[:-2] if len(core) > 2 else ""

    searches = [keyword]
    if place:
        searches.append(f"{place} {item}")
        searches += [f"{a} {item}" for a in _BILI_PLACE_ALIAS.get(place, [])]
    searches += _BILI_ITEM_ALIAS.get(core, [])
    searches = searches[:4]          # 并发上限 4 路，够覆盖，也不会拖慢页面

    names = [item] + _BILI_ITEM_ALIAS.get(core, [])
    scored, seen = [], set()
    # 并发检索：B站单次请求要 3–7 秒，串行会把页面拖到十几秒
    with ThreadPoolExecutor(max_workers=min(4, len(searches))) as pool:
        results = list(pool.map(_bili_search, searches))
    if not any(results):
        # 全部空（多半被限流）：冷却后重试一轮再放弃
        time.sleep(0.8)
        with ThreadPoolExecutor(max_workers=min(2, len(searches))) as pool:
            results = list(pool.map(_bili_search, searches[:2]))
    for videos in results:
        for v in videos:
            bvid = v.get("bvid")
            title = re.sub(r"<[^>]+>", "", v.get("title") or "").strip().replace("&amp;", "&")
            if not bvid or not title or bvid in seen:
                continue
            low = title.lower()
            typename = str(v.get("typename") or "")
            if any(b in typename for b in _BILI_BAD_TYPE):
                continue
            if any(b in low for b in _BILI_BAD_TITLE):
                continue
            # 必须真的在讲这件物品：标题要出现物品类名，且不能越过物品和领域红线
            title_flat = re.sub(r"\s+", "", title)
            if not any(n in title_flat for n in names):
                continue
            if any(b in low for b in _BILI_BAD_CONTEXT):
                continue
            # 游戏、音乐、舞蹈一律排除；影视里讲这个传说则保留
            if any(b in typename for b in _BILI_BAD_TYPE_STRICT):
                continue
            # 30 秒以内的多是片段/广告/循环短视频，讲不清一件物品
            secs = _bili_seconds(v.get("duration"))
            if secs and secs < 60:
                continue
            seen.add(bvid)
            cover = sum(1 for ch in set(core) if ch in title_flat)
            scored.append((cover, int(v.get("play") or 0), bvid, title, typename, secs))

    # 关键词覆盖度优先，其次播放量（热门的更可能是正经科普/纪录片）
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [
        {"bvid": s[2], "title": s[3], "play": s[1], "type": s[4], "secs": s[5]}
        for s in scored[:limit]
    ]


def _bili_pick(keyword, candidates):
    """让模型从候选里挑「确实在讲这件物品」的；一个都没有就返回 None。"""
    if not candidates:
        return None
    lines = chr(10).join(
        f"{i}. {c['title']}（分区：{c.get('type') or '未知'}，"
        f"时长 {c.get('secs', 0) // 60} 分，播放 {c['play']}）"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"我在为「{keyword}」寻找讲解视频。下面是B站搜索结果，每行是「序号. 标题（分区，播放量）」："
        + chr(10) + lines + chr(10) + chr(10)
        + f"请挑出标题确实在讲「{keyword}」这件物品或它的传说故事的编号"
        "（历史、来源、工艺、形制、文化、纪录片、非遗都算）。"
        "以下不算：同名的歌/舞/乐器演奏、游戏道具或攻略、动画角色、"
        "借这个词打比方（军事科技、搞笑段子）、卖货开箱测评、教程直播、"
        "与该物品无关的其它器物。"
        '只输出 JSON，不要解释：{"match": [编号, ...]}；一条都不相关就输出 {"match": []}。'
        "请先直接输出这个 JSON。"
    )
    try:
        r = requests.post(
            OFOX_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {OFOX_KEY}"},
            json={
                # 用非推理的小模型判定：推理模型在长候选表上要 10 秒以上，
                # 而且常把 max_tokens 耗在思考里、content 为空
                "model": os.environ.get(
                    "OFOX_JUDGE_MODEL", "google/gemini-2.5-flash-lite"
                ),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 200,
            },
            timeout=15,
        )
        msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or ""
        if not text.strip():  # 极端情况：只有思考过程，从里面兜出 JSON
            text = " ".join(
                d.get("text", "") for d in (msg.get("reasoning_details") or [])
            )
        parsed = _parse_json_loose(text) or {}
        picked = parsed.get("match")
        if not isinstance(picked, list):
            # 判定模型挂了/输出不可解析：不猜，宁可不放视频
            print(f"[search_bilibili] 判定输出不可解析：{str(text)[:120]}")
            return None
        ids = [i for i in picked if isinstance(i, int) and 0 <= i < len(candidates)]
        if not ids:
            return None
        # 候选已按「关键词覆盖度 → 播放量」排好序，取排名最靠前的匹配项
        return candidates[min(ids)]
    except Exception as e:
        print(f"[search_bilibili] 相关性判定失败：{e}")
        return None


@app.route("/api/search_bilibili", methods=["POST", "OPTIONS"])
def search_bilibili():
    """按 AI 提取的物品名搜 B站视频；搜不到真正相关的就不返回视频。"""
    blocked = _guard("bilibili")
    if blocked:
        return blocked

    data = request.get_json(silent=True) or {}
    keyword = (data.get("keyword") or "").strip()
    if not keyword:
        return jsonify({"error": "未找到相关视频"})

    try:
        best = _bili_pick(keyword, _bili_candidates(keyword))
        if not best:
            return jsonify({"error": "未找到相关视频"})
        return jsonify({"bvid": best["bvid"], "title": best["title"] or "B站视频"})
    except Exception as e:
        print(f"[search_bilibili] {keyword} 失败：{e}")
        return jsonify({"error": "未找到相关视频"})


if __name__ == "__main__":
    if not OFOX_KEY:
        print("⚠ 未检测到环境变量 OFOX_API_KEY，AI 问答将无法使用！")
    # PORT / HOST 由环境变量覆盖：本地默认只监听 127.0.0.1；
    # 部署到云平台时平台会注入 PORT，此时自动监听 0.0.0.0。
    _port = int(os.environ.get("PORT", "5000"))
    _host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print(f"文明信使已启动：http://{_host}:{_port}  （Ctrl+C 停止）")
    app.run(host=_host, port=_port, debug=False)
