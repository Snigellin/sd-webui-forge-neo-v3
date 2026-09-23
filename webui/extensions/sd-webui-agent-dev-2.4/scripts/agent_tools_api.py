# =============================================================================
# Agent Tools — 远程 API 图像生成/编辑工具
# =============================================================================

import base64
import json
import os
import time
import traceback
import urllib.request
import urllib.error
import urllib.parse
from io import BytesIO

from PIL import Image

from scripts.agent_config import load_config
from scripts.agent_tools_common import _apply_ui_aspect_size, _uploaded_image_size


def api_image_edit_tool(image, instruction, model=None, size="auto", response_format="b64_json"):
    """使用外部/API 图像模型编辑图片。用户明确指定 API 图像模型时优先使用。"""
    try:
        if image is None:
            return None, {"status": "error", "error": "请提供图片"}
        if not instruction or not str(instruction).strip():
            return None, {"status": "error", "error": "请提供图像编辑指令"}

        # 比例字符串 → 像素尺寸转换（auto/none/null 保持不变，让 API 自行决定）
        cfg = load_config()
        if str(cfg.get("image_aspect_ratio") or "").strip() == "参考图比例" and str(size or "").strip().lower() in ("", "auto", "none", "null"):
            original_size = _uploaded_image_size(image)
            size = f"{original_size[0]}x{original_size[1]}" if original_size else "1024x1024"
        else:
            size = _apply_ui_aspect_size(size, cfg)
        size = _resolve_image_size(size)
        print(f"[Agent] api_image_edit: size 参数解析为 {size}")
        # 当前 UI 选择是唯一模型来源，禁止 LLM 工具参数覆盖它。
        model_id = str(cfg.get("image_model") or "").strip()
        if not model_id:
            return None, {"status": "error", "error": "未指定 API 图像模型"}
        # 根据模型选择正确的 base URL（Gemini 模型使用专用端点）
        base_url = _get_image_api_base_url(cfg, model_id)
        api_key = _select_image_api_key(cfg)
        provider = str(cfg.get("image_api_provider") or "").strip().lower()
        if not base_url:
            return None, {"status": "error", "error": "未配置图像/视频生成 API Base URL"}
        if not api_key:
            return None, {"status": "error", "error": "未配置图像/视频生成 API Key，请在独立的生成 API 设置中填写，不是 Agent 大脑 API Key"}

        # Gemini 图像编辑模型使用 Google-native generateContent 格式
        if _is_gemini_image_model(model_id):
            # 将所有输入图片转为 base64。Gemini 原生协议支持在同一条
            # contents.parts 中放置多个 inlineData；不能只取第一张图，
            # 否则多图参考时后续图片会完全丢失。
            if not isinstance(image, (list, tuple)):
                image = [image]
            image_inputs = []
            for item in image:
                item_b64 = None
                item_mime = "image/png"
                if isinstance(item, Image.Image):
                    buf = BytesIO()
                    item.convert("RGB").save(buf, format="PNG")
                    item_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                elif isinstance(item, str) and os.path.isfile(item):
                    with open(item, "rb") as f:
                        item_b64 = base64.b64encode(f.read()).decode("utf-8")
                    if item.lower().endswith((".jpg", ".jpeg")):
                        item_mime = "image/jpeg"
                if item_b64:
                    image_inputs.append((item_b64, item_mime))

            if not image_inputs:
                return None, {"status": "error", "error": "无法读取输入图片用于 Gemini 编辑"}

            aspect_ratio, image_size = _pixels_to_gemini_config(size)
            images_b64, err = _call_gemini_generate(
                base_url, api_key, model_id, instruction, image_inputs,
                aspect_ratio=aspect_ratio, image_size=image_size,
            )
            if err:
                return None, err
            images = []
            for b64_data in images_b64:
                try:
                    img_data = base64.b64decode(b64_data)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                except Exception as e:
                    print(f"[Agent] Gemini 编辑图片解码失败: {e}")
            if images:
                info = {"status": "success", "model_used": model_id, "count": len(images)}
                return images, info
            return None, {"status": "error", "error": "Gemini 编辑图片解码失败"}

        if not isinstance(image, (list, tuple)):
            image = [image]

        images = []
        for item in image:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGBA"))
            elif isinstance(item, str) and os.path.isfile(item):
                images.append(Image.open(item).convert("RGBA"))
        if not images:
            return None, {"status": "error", "error": "image must be a PIL Image or file path"}

        image_b64_list = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_b64_list.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        # DashScope 图像编辑使用专用 multimodal-generation 协议。
        if provider == "dashscope":
            content = [{"text": str(instruction).strip()}]
            for image_b64 in image_b64_list[:5]:
                content.append({"image": f"data:image/png;base64,{image_b64}"})
            payload = {
                "model": model_id,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": content,
                    }],
                },
                "parameters": {"prompt_extend": True},
            }
            endpoint = base_url
            if endpoint.endswith("/v1"):
                endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                return None, {"status": "error", "error": f"DashScope 图像编辑失败: HTTP {e.code}", "detail": body, "model_used": model_id}

            images = []
            for choice in (result.get("output") or {}).get("choices") or []:
                for item in (choice.get("message") or {}).get("content") or []:
                    output = item.get("image")
                    if output:
                        if output.startswith("http"):
                            with urllib.request.urlopen(output, timeout=120) as image_resp:
                                images.append(Image.open(BytesIO(image_resp.read())).convert("RGB"))
                        else:
                            images.append(Image.open(BytesIO(base64.b64decode(output))).convert("RGB"))
            if not images:
                return None, {"status": "error", "error": "DashScope 未返回图像数据", "raw": result, "model_used": model_id}
            return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": "dashscope_multimodal_generation"}

        # YoboxAI 的 Gemini/banana 接口使用 generateContent 协议，
        # 图片必须放在 parts.inlineData，API Key 放在 query string。
        if _is_gemini_image_model(model_id) and "yoboxai.com" in base_url.lower():
            endpoint = (
                f"{base_url.rstrip('/')}/../gemini/v1beta/models/"
                f"{urllib.parse.quote(model_id, safe='')}:generateContent"
            )
            endpoint = endpoint.replace("/v1/../gemini/", "/gemini/")
            endpoint = f"{endpoint}?{urllib.parse.urlencode({'key': api_key})}"
            parts = [{"text": str(instruction).strip()}]
            for image_b64 in image_b64_list[:5]:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": image_b64,
                    }
                })
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": parts,
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            inferred_size = _infer_image_size_from_prompt(instruction, size)
            if inferred_size and str(inferred_size).lower() not in ("auto", "none", "null"):
                resolved_size = _resolve_image_size(inferred_size)
                aspect_ratio, image_size = _pixels_to_gemini_config(resolved_size)
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                }
                print(
                    f"[Agent] Gemini 编辑图像参数: aspectRatio={aspect_ratio}, "
                    f"imageSize={image_size}, sourceSize={resolved_size}"
                )
            else:
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": "1:1",
                    "imageSize": "1K",
                }

            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                return None, {
                    "status": "error",
                    "error": f"YoboxAI Gemini 图像编辑失败: HTTP {e.code}",
                    "detail": body,
                    "model_used": model_id,
                }

            images = []
            for candidate in result.get("candidates") or []:
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    inline_data = part.get("inlineData") or part.get("inline_data") or {}
                    output_b64 = inline_data.get("data")
                    if output_b64:
                        images.append(
                            Image.open(BytesIO(base64.b64decode(output_b64))).convert("RGB")
                        )
            if not images:
                return None, {
                    "status": "error",
                    "error": "YoboxAI Gemini 未返回图像数据",
                    "raw": result,
                    "model_used": model_id,
                }
            return images, {
                "status": "success",
                "model_used": model_id,
                "instruction": instruction,
                "method": "yoboxai_gemini_generate_content",
            }

        # ModelScope 的兼容层仍使用其专用的 generations 异步协议。
        # OpenAI 兼容的 gpt-image / sanye 编辑必须走真正的 /images/edits，
        # 不能把上传图片塞进 generations JSON，否则会退化成普通图像生成。
        if provider == "modelscope":
            payload = {
                "model": model_id,
                "prompt": str(instruction).strip(),
                "image_url": f"data:image/png;base64,{image_b64_list[0]}",
                "n": 1,
            }
            if size and str(size).lower() not in ("auto", "none", "null"):
                payload["size"] = size
            req = urllib.request.Request(
                f"{base_url}/images/generations",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
                },
            )
            if provider == "modelscope":
                req.headers["X-ModelScope-Async-Mode"] = "true"
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                label = "ModelScope" if provider == "modelscope" else provider or "API"
                return None, {"status": "error", "error": f"{label} 图像编辑失败: HTTP {e.code}", "detail": body, "model_used": model_id}

            # 异步模式：轮询任务结果
            if provider == "modelscope" and "task_id" in result:
                print(f"[Agent] ModelScope 编辑异步任务已提交: {result['task_id']}")
                images, err = _modelscope_poll_task(base_url, api_key, result["task_id"], "image_generation")
                if err:
                    return None, err
                return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": "modelscope_async_edit"}

            # 同步模式回退
            data_items = result.get("data") or []
            if not data_items:
                return None, {"status": "error", "error": "ModelScope 未返回图像数据", "raw": result, "model_used": model_id}
            images = []
            for item in data_items:
                b64 = item.get("b64_json") or item.get("image")
                if b64:
                    if isinstance(b64, str) and b64.startswith("data:image"):
                        b64 = b64.split(",", 1)[1]
                    images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
                elif item.get("url"):
                    with urllib.request.urlopen(item["url"], timeout=120) as img_resp:
                        images.append(Image.open(BytesIO(img_resp.read())).convert("RGB"))
            if not images:
                return None, {"status": "error", "error": "ModelScope 编辑未返回可解析图像", "raw": result, "model_used": model_id}
            return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": f"{provider or 'api'}_generations_edit"}

        # OpenAI-compatible图像编辑接口要求 multipart/form-data，不能把 image
        # 作为 data URL 放进 JSON。
        boundary = f"----ForgeAgent{int(time.time() * 1000000)}"
        parts = []

        def add_text(name, value):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
            )

        add_text("model", model_id)
        add_text("prompt", str(instruction).strip())
        add_text("response_format", response_format)
        if size and str(size).lower() not in ("auto", "none", "null"):
            add_text("size", size)
        for idx, image_b64 in enumerate(image_b64_list[:5]):
            png_bytes = base64.b64decode(image_b64)
            # OpenAI Images Edits 使用 image 字段；多张图时重复 image 字段。
            # image0/image1 等字段名会被多数兼容服务忽略，导致“编辑无输入图”。
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input_{idx}.png\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8")
                + png_bytes + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode("ascii"))
        data = b"".join(parts)
        url = f"{base_url}/images/edits"
        edit_headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {api_key}",
        }
        # ModelScope 使用异步模式
        is_modelscope = provider == "modelscope"
        if is_modelscope:
            edit_headers["X-ModelScope-Async-Mode"] = "true"
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=edit_headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return None, _format_api_http_error(e, "图像编辑")

        # ModelScope 异步模式：轮询任务结果
        if is_modelscope and "task_id" in result:
            print(f"[Agent] ModelScope 编辑异步任务已提交: {result['task_id']}")
            images, err = _modelscope_poll_task(base_url, api_key, result["task_id"], "image_generation")
            if err:
                return None, err
            return images, {"status": "success", "model_used": model_id, "instruction": instruction, "method": "modelscope_async_edit"}

        data_items = result.get("data") or []
        if not data_items:
            return None, {"status": "error", "error": "API 未返回图像数据", "raw": result, "model_used": model_id}

        images = []
        for item in data_items:
            b64 = item.get("b64_json") or item.get("image")
            if b64:
                if isinstance(b64, str) and b64.startswith("data:image"):
                    b64 = b64.split(",", 1)[1]
                images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
            elif item.get("url"):
                try:
                    with urllib.request.urlopen(item["url"], timeout=120) as image_resp:
                        images.append(Image.open(BytesIO(image_resp.read())).convert("RGB"))
                except Exception as e:
                    return None, {"status": "error", "error": f"API 返回图片 URL 无法读取: {e}", "model_used": model_id}

        if not images:
            return None, {"status": "error", "error": "API 返回中没有可解析的 base64 图像", "raw": result, "model_used": model_id}

        return images, {
            "status": "success",
            "model_used": model_id,
            "instruction": instruction,
            "method": "api_image_edit",
        }
    except Exception as e:
        return None, {"status": "error", "error": str(e), "method": "api_image_edit"}


def _modelscope_poll_task(base_url, api_key, task_id, task_type="image_generation", max_polls=60, poll_interval=3):
    """轮询 ModelScope 异步任务，返回 (images_list, error_dict_or_None)。"""
    for i in range(max_polls):
        time.sleep(poll_interval)
        poll_req = urllib.request.Request(
            f"{base_url}/tasks/{task_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-ModelScope-Task-Type": task_type,
            },
        )
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as poll_resp:
                poll_result = json.loads(poll_resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:500]
            return None, {"status": "error", "error": f"ModelScope 轮询失败: HTTP {e.code}", "detail": body}
        except Exception as e:
            # 网络超时等可恢复错误，继续轮询
            print(f"[Agent] ModelScope 轮询 {i+1} 次异常（继续重试）: {e}")
            continue

        status = str(poll_result.get("task_status") or poll_result.get("status", "")).upper()
        print(f"[Agent] ModelScope 任务轮询 {i+1}/{max_polls}: status={status}")

        if status in ("SUCCEEDED", "SUCCEED", "COMPLETED"):
            images = []
            # ModelScope 返回 output_images 数组
            output_images = poll_result.get("output_images") or []
            for img_url in output_images:
                if isinstance(img_url, str) and img_url.startswith("http"):
                    try:
                        with urllib.request.urlopen(img_url, timeout=60) as img_resp:
                            images.append(Image.open(BytesIO(img_resp.read())).convert("RGB"))
                    except Exception as e:
                        print(f"[Agent] ModelScope 下载图片失败: {e}")
            if images:
                return images, None
            return None, {"status": "error", "error": "ModelScope 任务成功但未返回图片", "raw": poll_result}
        elif status in ("FAILED", "FAILURE", "ERROR"):
            err_msg = poll_result.get("errors") or poll_result.get("error") or "未知错误"
            return None, {"status": "error", "error": f"ModelScope 任务失败: {err_msg}", "raw": poll_result}

    return None, {"status": "error", "error": f"ModelScope 任务轮询超时（{max_polls * poll_interval}秒）"}


_ASPECT_RATIO_TO_PIXELS = {
    "1:1": "1024x1024",
    "9:16": "1024x1792",
    "16:9": "1792x1024",
    "3:4": "768x1024",
    "4:3": "1024x768",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "21:9": "1792x768",
}


def _resolve_image_size(size):
    """将 size 参数统一解析为 API 可接受的像素尺寸字符串。

    - 比例格式 (如 '9:16') → 映射为像素尺寸 (如 '1024x1792')
    - 像素格式 (如 '1024x1792') → 原样返回
    - 空值/无效值 → 默认 '1024x1024'
    """
    if not size or not str(size).strip():
        return "1024x1024"
    s = str(size).strip().lower()
    # 比例格式（含冒号）
    if ":" in s:
        mapped = _ASPECT_RATIO_TO_PIXELS.get(s)
        if mapped:
            return mapped
        # 未知比例，回退到默认
        print(f"[Agent] 未知比例 '{s}'，回退到 1024x1024")
        return "1024x1024"
    # 像素格式（含 x）
    if "x" in s:
        return s
    # 其他情况回退默认
    print(f"[Agent] 无法识别的 size '{s}'，回退到 1024x1024")
    return "1024x1024"


def _pixels_to_gemini_config(size):
    """将统一的像素尺寸转换为 Gemini imageConfig 参数。"""
    config = {
        "1024x1024": ("1:1", "1K"),
        "1024x1792": ("9:16", "2K"),
        "1792x1024": ("16:9", "2K"),
        "768x1024": ("3:4", "1K"),
        "1024x768": ("4:3", "1K"),
        "832x1248": ("2:3", "1K"),
        "1248x832": ("3:2", "1K"),
        "1792x768": ("21:9", "2K"),
    }
    return config.get(str(size).lower(), ("1:1", "1K"))


def _infer_image_size_from_prompt(prompt, size):
    """当模型漏传 size 时，从用户提示词中的比例描述兜底推断尺寸。

    注意：英文 "portrait" 不在此列——它在生图提示词里多是题材词
    （portrait = 肖像/头像，如 "girl portrait avatar"），会误判成竖版；
    方向判断只信明确的 "vertical" / "9:16" / 中文"竖版、竖屏、手机壁纸"。
    """
    text = str(prompt or "").lower()
    current = str(size or "").strip().lower()
    if current not in ("", "auto", "none", "null", "1024x1024", "1:1"):
        return size
    if any(token in text for token in ("16:9", "横版", "横屏", "宽幅", "wide", "landscape")):
        return "1792x1024"
    if any(token in text for token in ("9:16", "竖版", "竖屏", "手机壁纸", "vertical")):
        return "1024x1792"
    if "4:3" in text:
        return "1024x768"
    if "3:4" in text:
        return "768x1024"
    return size


def api_image_generate_tool(prompt, model=None, size="1024x1024", response_format="b64_json", negative_prompt="", quality=""):
    """使用设置区选择的外部/API 图像模型生成图片。

    参数:
        prompt: 正向提示词
        model: API 模型 ID（为空则用配置中的 image_model）
        size: 输出尺寸，如 1024x1024、16:9 等
        response_format: b64_json 或 url
        negative_prompt: 负向提示词（YoboxAI gpt-image-2 支持）
        quality: 质量档位，如 high/medium/low（YoboxAI gpt-image-2 支持）
    """
    try:
        if not prompt or not str(prompt).strip():
            return None, {"status": "error", "error": "请提供图像生成提示词"}

        # 比例字符串 → 像素尺寸转换（gpt-image-2 等 OpenAI 兼容接口只接受像素尺寸）
        cfg = load_config()
        size = _infer_image_size_from_prompt(prompt, size)
        prompt_text = str(prompt or "").lower()
        prompt_sets_ratio = any(token in prompt_text for token in (
            "16:9", "9:16", "1:1", "4:3", "3:4", "横版", "横屏", "竖版", "竖屏",
            "手机壁纸", "桌面壁纸", "square", "wide", "landscape", "vertical", "portrait",
        ))
        if not prompt_sets_ratio:
            size = _apply_ui_aspect_size(size, cfg, allow_auto=False)
        size = _resolve_image_size(size)
        print(f"[Agent] api_image_generate: size 参数解析为 {size}")

        # 当前 UI 选择是唯一模型来源，禁止 LLM 工具参数覆盖它。
        model_id = str(cfg.get("image_model") or "").strip()
        # 根据模型选择正确的 base URL（Gemini 模型使用专用端点）
        base_url = _get_image_api_base_url(cfg, model_id)
        # 根据供应商选择正确的 key：避免跨供应商混用导致 401
        api_key = _select_image_api_key(cfg)
        provider = str(cfg.get("image_api_provider") or "").strip().lower()
        # 诊断日志
        _kp = cfg.get("image_api_key") or ""
        _kp2 = cfg.get("api_key") or ""
        _k_mask = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else ("(空)" if not api_key else "***")
        _key_source = "LLM_key" if (api_key == _kp2 and _kp2) else ("image_key" if api_key == _kp and _kp else "fallback")
        print(f"[Agent] api_image_generate: model={model_id}, provider={provider}, base_url={base_url}, api_key={_k_mask} (image_key={'有' if _kp else '无'}, LLM_key={'有' if _kp2 else '无'}, key_source={_key_source})")
        if not model_id:
            return None, {"status": "error", "error": "未指定 API 图像模型"}
        if not base_url:
            return None, {"status": "error", "error": "未配置生成 API Base URL"}
        if not api_key:
            return None, {"status": "error", "error": "未配置生成 API Key，请在生成 API 设置中填写"}

        # Gemini 图像模型使用 Google-native generateContent 格式
        if _is_gemini_image_model(model_id):
            aspect_ratio, image_size = _pixels_to_gemini_config(size)
            print(f"[Agent] Gemini 图像参数: aspectRatio={aspect_ratio}, imageSize={image_size}")
            images_b64, err = _call_gemini_generate(
                base_url,
                api_key,
                model_id,
                prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            )
            if err:
                return None, err
            # 将 base64 转为 PIL Image
            images = []
            for b64_data in images_b64:
                try:
                    img_data = base64.b64decode(b64_data)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                except Exception as e:
                    print(f"[Agent] Gemini 图片解码失败: {e}")
            if images:
                info = {"status": "success", "model_used": model_id, "count": len(images)}
                return images, info
            return None, {"status": "error", "error": "Gemini 图片解码失败"}

        if provider == "yoboxai" and _is_gemini_image_model(model_id):
            gemini_model_id = model_id
            endpoint = (
                f"{base_url.rstrip('/')}/../gemini/v1beta/models/"
                f"{urllib.parse.quote(gemini_model_id, safe='')}:generateContent"
            )
            endpoint = endpoint.replace("/v1/../gemini/", "/gemini/")
            endpoint = f"{endpoint}?{urllib.parse.urlencode({'key': api_key})}"
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": str(prompt).strip()}],
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:1200]
                return None, {"status": "error", "error": f"YoboxAI Gemini 图像生成失败: HTTP {e.code}", "detail": body, "model_used": model_id}

            images = []
            for candidate in result.get("candidates") or []:
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    inline_data = part.get("inlineData") or part.get("inline_data") or {}
                    output_b64 = inline_data.get("data")
                    if output_b64:
                        images.append(Image.open(BytesIO(base64.b64decode(output_b64))).convert("RGB"))
            if not images:
                return None, {"status": "error", "error": "YoboxAI Gemini 未返回图像数据", "raw": result, "model_used": model_id}
            return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "yoboxai_gemini_generate_content"}

        # 非 OpenAI 兼容供应商：专用适配
        if provider == "dashscope":
            return _dashscope_generate(base_url, api_key, model_id, prompt, size)
        if provider == "stability":
            return _stability_generate(base_url, api_key, model_id, prompt, size)
        if provider == "minimax":
            return _minimax_generate(base_url, api_key, model_id, prompt, size, negative_prompt=negative_prompt)
        if provider == "ideogram":
            return _ideogram_generate(base_url, api_key, model_id, prompt, size)

        payload = {
            "model": model_id,
            "prompt": str(prompt).strip(),
            "n": 1,
            "size": size,
        }
        # YoboxAI gpt-image-2 支持扩展参数
        if negative_prompt and str(negative_prompt).strip():
            payload["negative_prompt"] = str(negative_prompt).strip()
        if quality and str(quality).strip():
            payload["quality"] = str(quality).strip()
        if response_format:
            payload["response_format"] = response_format
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # ModelScope 使用异步模式
        is_modelscope = provider == "modelscope"
        if is_modelscope:
            request_headers["X-ModelScope-Async-Mode"] = "true"
        req = urllib.request.Request(
            f"{base_url}/images/generations",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=request_headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return None, _format_api_http_error(e, "图像生成")

        # ModelScope 异步模式：轮询任务结果
        if is_modelscope and "task_id" in result:
            print(f"[Agent] ModelScope 异步任务已提交: {result['task_id']}")
            images, err = _modelscope_poll_task(base_url, api_key, result["task_id"], "image_generation")
            if err:
                return None, err
            return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "modelscope_async_generate"}

        images = []
        for item in result.get("data") or []:
            b64 = item.get("b64_json") or item.get("image")
            if b64:
                if isinstance(b64, str) and b64.startswith("data:image"):
                    b64 = b64.split(",", 1)[1]
                images.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
            elif item.get("url"):
                with urllib.request.urlopen(item["url"], timeout=120) as image_resp:
                    images.append(Image.open(BytesIO(image_resp.read())).convert("RGB"))
        if not images:
            return None, {"status": "error", "error": "API 未返回图像数据", "raw": result, "model_used": model_id}
        return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "api_image_generate"}
    except Exception as e:
        print(f"[Agent] api_image_generate_tool 异常: {traceback.format_exc()}")
        return None, {"status": "error", "error": str(e), "method": "api_image_generate"}


def _select_image_api_key(cfg):
    """根据图像供应商选择正确的 API Key，避免跨供应商 key 混用导致 401。

    逻辑：
    - 如果图像供应商 == LLM 供应商（如都是 YoboxAI），使用 LLM 的 api_key
    - 否则使用 image_api_key
    - 如果 image_api_key 为空，回退到 api_key
    """
    image_provider = str(cfg.get("image_api_provider") or "").strip().lower()
    llm_provider = str(cfg.get("api_provider") or "").strip().lower()
    image_key = str(cfg.get("image_api_key") or "").strip()
    llm_key = str(cfg.get("api_key") or "").strip()

    if image_provider and image_provider == llm_provider:
        # 同一供应商，优先用 LLM key
        return llm_key or image_key
    # 不同供应商，用图像专用 key；如果为空则回退到 LLM key
    return image_key or llm_key


def _get_image_api_base_url(cfg, model_id=""):
    """根据供应商与模型 ID 选择正确的 API Base URL。

    YoboxAI 的 Gemini/banana 图像模型需要专用端点 https://api.yoboxai.com/gemini；
    其他供应商（如官方 Gemini）使用各自设置里配置的 base_url。
    """
    model_lower = str(model_id or "").lower()
    provider = str(cfg.get("image_api_provider") or "").strip().lower()
    base_url = str(cfg.get("image_base_url") or "").rstrip("/")
    # 用户可能直接粘贴文档中的完整接口地址；内部统一保留 OpenAI 兼容根地址，
    # 下面的请求再追加 /images/generations。
    for suffix in ("/images/generations", "/chat/completions"):
        if base_url.lower().endswith(suffix):
            base_url = base_url[: -len(suffix)].rstrip("/")
            break
    if (model_lower.startswith("gemini") or model_lower.startswith("banana")) and (
        provider == "yoboxai" or "yoboxai.com" in base_url.lower() or not base_url
    ):
        return "https://api.yoboxai.com/gemini"
    return base_url


def _is_gemini_image_model(model_id):
    """判断是否为 Gemini 图像模型（需要 Google-native generateContent 格式）。

    banana2/bananapro 是 YoboxAI 对 Gemini 图像模型的别名，也需要走 Gemini 端点。
    """
    model_lower = str(model_id or "").lower()
    return model_lower.startswith("gemini") or model_lower.startswith("banana")


def _call_gemini_generate(base_url, api_key, model_id, prompt_text, image_b64=None, image_mime="image/png",
                          aspect_ratio="1:1", image_size="1K"):
    """调用 Gemini generateContent API 生成/编辑图片。

    返回 (images_list, error_dict)，成功时 images_list 为 base64 图片列表，error_dict 为 None。
    支持 imageConfig: aspectRatio (1:1, 16:9, 9:16, 4:3, 3:4), imageSize (1K, 2K, 4K)。
    """
    parts = []
    if image_b64:
        # 兼容旧的单图参数，同时支持 [(base64, mime), ...] 多图参数。
        if isinstance(image_b64, (list, tuple)):
            image_items = image_b64
        else:
            image_items = [(image_b64, image_mime)]
        for item in image_items:
            if isinstance(item, (list, tuple)):
                item_data = item[0] if item else None
                item_mime = item[1] if len(item) > 1 else "image/png"
            else:
                item_data = item
                item_mime = image_mime
            if item_data:
                parts.append({
                    "inlineData": {
                        "mimeType": item_mime,
                        "data": item_data,
                    }
                })
    parts.append({"text": str(prompt_text).strip()})

    payload = {
        "contents": [{
            "role": "user",
            "parts": parts,
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio or "1:1",
                "imageSize": image_size or "1K",
            },
        },
    }

    # 认证方式：?key= 查询参数（官方 Gemini 与 YoboxAI Gemini 端点都接受）
    base = base_url.rstrip("/")
    if base.endswith("/v1beta"):
        endpoint = f"{base}/models/{urllib.parse.quote(model_id, safe='')}:generateContent?{urllib.parse.urlencode({'key': api_key})}"
    else:
        endpoint = f"{base}/v1beta/models/{model_id}:generateContent?key={api_key}"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return None, _format_api_http_error(e, "Gemini 图像生成")
    except Exception as e:
        return None, {"status": "error", "error": f"Gemini API 调用异常: {e}"}

    # 解析响应，提取 inlineData 图片
    images = []
    try:
        candidates = result.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])
            for part in parts_list:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    images.append(inline["data"])
    except Exception as e:
        return None, {"status": "error", "error": f"解析 Gemini 响应失败: {e}", "detail": str(result)[:500]}

    if not images:
        # 没有返回图片，可能只返回了文本
        text_parts = []
        try:
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("text"):
                        text_parts.append(part["text"])
        except Exception:
            pass
        text_info = " ".join(text_parts)[:200] if text_parts else "无内容"
        return None, {"status": "error", "error": f"Gemini 未返回图片，仅返回文本: {text_info}"}

    return images, None


def _format_api_http_error(e, action="调用"):
    """格式化 API HTTP 错误，特别是 401 鉴权失败时给出明确提示，防止 LLM 静默回退本地。"""
    body = ""
    try:
        body = e.read().decode("utf-8", errors="ignore")[:1200]
    except Exception:
        pass
    if e.code == 401:
        return {
            "status": "error",
            "error": f"API {action}鉴权失败 (HTTP 401)：当前选择的 API Key 与供应商不匹配或已失效。请检查图像生成 API 设置中的供应商和 Key 是否对应。不要切换到本地模型，应直接告知用户此错误。",
            "detail": body,
            "http_code": 401,
            "do_not_fallback": True,
        }
    if e.code == 403:
        if "1010" in body:
            return {
                "status": "error",
                "error": f"API {action}被供应商网络防火墙拦截 (HTTP 403 / Cloudflare 1010)，不是模型选择错误。请检查 sanye 是否允许当前访问来源调用图像接口。",
                "detail": body,
                "http_code": 403,
                "do_not_fallback": True,
            }
        return {
            "status": "error",
            "error": f"API {action}权限不足 (HTTP 403)：当前 Key 无权限访问该模型。不要切换到本地模型，应直接告知用户此错误。",
            "detail": body,
            "http_code": 403,
            "do_not_fallback": True,
        }
    return {
        "status": "error",
        "error": f"API {action}失败: HTTP {e.code}",
        "detail": body,
        "http_code": e.code,
    }


# =============================================================================
# 非 OpenAI 兼容供应商专用适配（DashScope / Stability / MiniMax / Ideogram / Gemini）
# 参照 Forge 核心 modules_forge/api_providers.py 的同名实现移植
# =============================================================================

def _import_requests():
    try:
        import requests
        return requests
    except Exception as e:
        raise RuntimeError(f"缺少 requests 依赖，无法调用该供应商（请安装 requirements.txt）: {e}")


def _api_post_json(url, headers, payload, timeout=180):
    requests = _import_requests()
    resp = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"响应解析失败: {e}: {resp.text[:300]}")


def _response_to_image(data, is_b64=False):
    if data is None:
        raise RuntimeError("供应商未返回图像数据")
    value = str(data).strip()
    if value.startswith("data:image"):
        value = value.split(",", 1)[-1]
        is_b64 = True
    if not is_b64 and value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=120) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGB")
    return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")


def _dashscope_generate(base_url, api_key, model_id, prompt, size, count=1):
    """阿里云百炼（DashScope）多模态生图：output.choices[].message.content[].image"""
    payload = {
        "model": model_id,
        "input": {"messages": [{"role": "user", "content": [{"text": str(prompt).strip()}]}]},
        "parameters": {"size": str(size).replace("x", "*"), "n": max(1, int(count or 1))},
    }
    result = _api_post_json(
        base_url,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
    )
    images = []
    for choice in ((result.get("output") or {}).get("choices") or []):
        for item in ((choice.get("message") or {}).get("content") or []):
            if item.get("image"):
                images.append(_response_to_image(item["image"], is_b64=not str(item["image"]).startswith("http")))
    if not images:
        raise RuntimeError(f"DashScope 未返回图片: {str(result)[:300]}")
    return images, {"status": "success", "model_used": model_id, "prompt": prompt, "method": "dashscope_multimodal_generation"}


def _stability_generate(base_url, api_key, model_id, prompt, size, count=1):
    """Stability AI v2beta：multipart 表单，响应为原始图片字节（Accept: image/*）"""
    requests = _import_requests()
    model = str(model_id or "ultra").strip() or "ultra"
    aspect_ratio, _ = _pixels_to_gemini_config(size)
    url = f"{base_url.rstrip('/')}/stable-image/generate/{model}"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    form = {"prompt": str(prompt).strip(), "aspect_ratio": aspect_ratio, "output_format": "png"}
    images = []
    for _ in range(max(1, int(count or 1))):
        resp = requests.post(url, headers=headers, files={"none": (None, "")}, data=form, timeout=180)
        if resp.status_code >= 400:
            raise RuntimeError(f"Stability API HTTP {resp.status_code}: {resp.text[:300]}")
        images.append(Image.open(BytesIO(resp.content)).convert("RGB"))
    return images, {"status": "success", "model_used": model, "prompt": prompt, "method": "stability_generate"}


def _minimax_generate(base_url, api_key, model_id, prompt, size, count=1, negative_prompt=""):
    """MiniMax 生图：data.image_urls / data.image_base64"""
    aspect_ratio, _ = _pixels_to_gemini_config(size)
    payload = {
        "model": str(model_id or "image-01").strip() or "image-01",
        "prompt": str(prompt).strip(),
        "aspect_ratio": aspect_ratio,
        "n": max(1, int(count or 1)),
        "response_format": "url",
    }
    result = _api_post_json(
        f"{base_url.rstrip('/')}/image_generation",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
    )
    data = result.get("data") or {}
    images = []
    urls = data.get("image_urls") or ([data["image_url"]] if data.get("image_url") else [])
    for url in urls:
        images.append(_response_to_image(url))
    for b64 in (data.get("image_base64") or []):
        images.append(_response_to_image(b64, is_b64=True))
    if not images:
        raise RuntimeError(f"MiniMax 未返回图片: {str(result)[:300]}")
    return images, {"status": "success", "model_used": payload["model"], "prompt": prompt, "method": "minimax_image_generation"}


def _ideogram_generate(base_url, api_key, model_id, prompt, size, count=1):
    """Ideogram：POST /generate，data[].data[].base64 / url"""
    aspect_ratio, _ = _pixels_to_gemini_config(size)
    headers = {
        "Api-Key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    images = []
    for _ in range(max(1, int(count or 1))):
        payload = {
            "image_request": {
                "prompt": str(prompt).strip(),
                "model": str(model_id or "V_2").strip() or "V_2",
                "aspect_ratio": aspect_ratio,
            }
        }
        result = _api_post_json(f"{base_url.rstrip('/')}/generate", headers, payload)
        for block in (result.get("data") or []):
            for item in (block.get("data") or []):
                if item.get("base64"):
                    images.append(_response_to_image(item["base64"], is_b64=True))
                elif item.get("url"):
                    images.append(_response_to_image(item["url"]))
    if not images:
        raise RuntimeError("Ideogram 未返回图片")
    return images, {"status": "success", "model_used": str(model_id or "V_2"), "prompt": prompt, "method": "ideogram_generate"}
