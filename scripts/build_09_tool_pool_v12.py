#!/usr/bin/env python3
"""
Build pages/09-tool-pool-v12.html — 3-model comparison + baseline
Reads: s4_tool_registry_v12.json, s2_case_abilities_116.json,
       121case-gemini_3.1or3.0-tool_v10orv12.jsonl,
       s5_agent_results_tool_v12_gpt-5.4.jsonl,
       s5_agent_results_tool_v12_claude-opus-4-6.jsonl,
       s5_agent_results_direct_gemini-3.1-pro.jsonl (baseline),
       execution-comparison-v2.json
Output: self-contained HTML with embedded data
"""
import json
import os
import glob
import re
import base64
from collections import Counter, defaultdict
from io import BytesIO

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not installed, image embedding disabled")

BASE = "/root/paddlejob/workspace/env_run/output/bwh/lj"
DATA_DIR = os.path.join(BASE, "video_cases_tools/data/outputs")
VIEWER_DIR = os.path.join(BASE, "lv-viewer")

# Canonical model order for display
MODEL_ORDER = ["gpt-5.4", "claude-opus-4-6", "gemini"]

# ── Ability → Category mapping ──
ABILITY_CATEGORY = {
    1: "High-Level Reasoning", 2: "High-Level Reasoning", 3: "High-Level Reasoning",
    4: "High-Level Reasoning", 5: "High-Level Reasoning",
    6: "Temporal Relation", 7: "Temporal Relation", 8: "Temporal Relation",
    9: "Temporal Relation", 10: "Temporal Relation",
    11: "Spatial Relation", 12: "Spatial Relation", 13: "Spatial Relation",
    14: "Spatial Relation",
    15: "Audio-Visual", 16: "Audio-Visual", 17: "Audio-Visual",
    18: "Visual Perception", 19: "Visual Perception", 20: "Visual Perception",
    21: "Visual Perception", 22: "Visual Perception",
}


# ── Key cases for image visualization ──
# Cat1: BL✓ All-Tool✗ | Cat3: BL✗ All-Tool✓
KEY_VISUAL_CASES = {
    "longvideobench_hf4WUOagFAw_1",
    "lvbench_1069", "lvbench_1135", "lvbench_2362",
    "lvbench_4023", "lvbench_4754",
    "longvideobench_4QSmRYQBfN4_0", "videomme_011-3",
    # Cat2: BL✓ + 2 Tools✗
    "videomme_053-1", "videomme_416-1", "lvbench_63",
    "lvbench_135", "videomme_019-2", "videomme_619-2",
    "lvbench_61", "lvbench_41",
}

# Visual tool types that produce images/bboxes
VISUAL_TOOLS = {"frame_extraction", "object_detection", "spatial_crop"}

MAX_FRAMES_PER_STEP = 6  # Max frames to embed per frame_extraction step
IMG_MAX_WIDTH = 280       # Thumbnail width in pixels
IMG_JPEG_QUALITY = 55     # JPEG quality

# Output directory for generated images (relative to lv-viewer root)
IMG_OUTPUT_DIR = os.path.join(VIEWER_DIR, "images", "tool-visuals")
IMG_REL_PREFIX = "../images/tool-visuals"  # Relative path from pages/ to images


_img_counter = [0]  # mutable counter for unique filenames


def _save_img(img, case_id, model_key, suffix):
    """Save PIL Image to file and return relative path from pages/."""
    _img_counter[0] += 1
    case_dir = os.path.join(IMG_OUTPUT_DIR, case_id, model_key)
    os.makedirs(case_dir, exist_ok=True)
    fname = f"{_img_counter[0]:04d}_{suffix}.jpg"
    out_path = os.path.join(case_dir, fname)
    img.save(out_path, format="JPEG", quality=IMG_JPEG_QUALITY)
    return f"{IMG_REL_PREFIX}/{case_id}/{model_key}/{fname}"


def img_to_file(img_path, case_id, model_key, suffix="frame", max_w=IMG_MAX_WIDTH):
    """Resize image and save to output dir. Return relative path."""
    if not HAS_PIL or not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        return _save_img(img, case_id, model_key, suffix)
    except Exception:
        return None


def draw_bboxes_to_file(img_path, detections, case_id, model_key, suffix="bbox", max_w=IMG_MAX_WIDTH):
    """Draw bounding boxes on image, save to file, return relative path."""
    if not HAS_PIL or not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = ["#FF3333", "#33FF33", "#3333FF", "#FF33FF", "#FFFF33", "#33FFFF"]
        for i, det in enumerate(detections):
            bbox = det.get("bbox") or det.get("bounding_box", [])
            if len(bbox) != 4:
                continue
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = [float(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 300))
            label = det.get("object", "")
            if label:
                draw.text((x1 + 2, y1 + 2), label, fill=color)
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        return _save_img(img, case_id, model_key, suffix)
    except Exception:
        return None


def extract_visual_data_for_step(tool_data, tool_name, case_id, model_key, step_num, frame_lookup=None):
    """Extract visual data from a tool output step, save images to files.
    Returns list of {src: relative_path, label: str} or None.
    """
    if not HAS_PIL:
        return None
    content = tool_data.get("content", "")
    if isinstance(content, list):
        text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    else:
        text = str(content)

    images = []

    if tool_name == "frame_extraction" and "image_path" in text:
        try:
            data = json.loads(text) if text.strip().startswith("{") else json.loads(
                text.strip("` \n").lstrip("json\n"))
            frames = data.get("frames", [])
            if len(frames) > MAX_FRAMES_PER_STEP:
                step_size = len(frames) / MAX_FRAMES_PER_STEP
                indices = [int(i * step_size) for i in range(MAX_FRAMES_PER_STEP)]
                frames = [frames[i] for i in indices]
            for fi, fr in enumerate(frames):
                path = fr.get("image_path", "")
                ts = fr.get("timestamp", 0)
                rel = img_to_file(path, case_id, model_key, suffix=f"s{step_num}_f{fi}_{ts:.1f}s")
                if rel:
                    images.append({"src": rel, "label": f"{ts:.1f}s"})
        except Exception:
            pass

    elif tool_name == "object_detection" and ("bbox" in text or "bounding_box" in text):
        try:
            data = json.loads(text) if text.strip().startswith("{") else json.loads(
                text.strip("` \n").lstrip("json\n"))
            dets = data.get("detections", [])
            if dets:
                input_args = tool_data.get("input_args", {})
                img_path = None
                vf = input_args.get("video_frames", "")
                if isinstance(vf, str) and os.path.exists(vf):
                    img_path = vf
                elif isinstance(vf, list) and vf and os.path.exists(str(vf[0])):
                    img_path = str(vf[0])
                if not img_path and frame_lookup:
                    fi = input_args.get("frame_index") or (input_args.get("frame_indices", [None])[0] if input_args.get("frame_indices") else None)
                    if fi is not None and fi in frame_lookup:
                        img_path = frame_lookup[fi]
                if img_path:
                    rel = draw_bboxes_to_file(img_path, dets, case_id, model_key, suffix=f"s{step_num}_bbox")
                    if rel:
                        labels = [f"{d.get('object', '?')}" for d in dets[:5]]
                        images.append({"src": rel, "label": ", ".join(labels)})
        except Exception:
            pass

    elif tool_name == "spatial_crop" and "crop_path" in text:
        try:
            data = json.loads(text) if text.strip().startswith("{") else json.loads(
                text.strip("` \n").lstrip("json\n"))
            crop_path = data.get("crop_path", "")
            bbox = data.get("bbox_pixel", [])
            label = f"crop [{','.join(str(int(v)) for v in bbox)}]" if bbox else "crop"
            rel = img_to_file(crop_path, case_id, model_key, suffix=f"s{step_num}_crop")
            if rel:
                images.append({"src": rel, "label": label})
        except Exception:
            pass

    return images if images else None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    """Load JSONL, skip _meta lines."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                continue
            data.append(d)
    return data


def strip_prefix(qid):
    for prefix in ("longvideobench_", "lvbench_", "videomme_"):
        if qid.startswith(prefix):
            return qid[len(prefix):]
    return qid


def _extract_tool_output_text(content):
    """Extract text from tool output content (str or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def parse_trajectory_steps(trajectory, traj_folder=None, embed_images=False, case_id="", model_key=""):
    """Parse trajectory messages into steps with tool args and output.
    If traj_folder is provided, load full data (including thinking) from step files.
    If embed_images=True, save visual data to files for key cases.
    """
    # If trajectory folder exists, prefer loading from files (has thinking + full content)
    if traj_folder and os.path.exists(traj_folder):
        return _parse_from_folder(traj_folder, embed_images=embed_images,
                                   case_id=case_id, model_key=model_key)

    steps = []
    step_num = 0
    pending_tool_calls = []
    tool_output_queue = []

    # First pass: pair AI tool_calls with subsequent tool outputs
    paired = []  # list of ("tool_call"|"reasoning", ai_content, thinking, [(tool_name, args, output), ...])
    i = 0
    while i < len(trajectory):
        item = trajectory[i]
        if item.get("role") == "ai":
            content = (item.get("content") or "")
            thinking = (item.get("thinking") or "")
            tool_calls = item.get("tool_calls") or []
            if tool_calls:
                tools_with_output = []
                for tc in tool_calls:
                    output_text = ""
                    j = i + 1
                    while j < len(trajectory):
                        if trajectory[j].get("role") == "tool":
                            output_text = _extract_tool_output_text(trajectory[j].get("content", ""))
                            j += 1
                            break
                        elif trajectory[j].get("role") == "ai":
                            break
                        j += 1
                    tools_with_output.append((tc.get("name", ""), tc.get("args", {}), output_text))
                paired.append(("tool_call", content, thinking, tools_with_output))
                i += 1
                while i < len(trajectory) and trajectory[i].get("role") == "tool":
                    i += 1
                continue
            else:
                paired.append(("reasoning", content, thinking, []))
        i += 1

    # Build steps
    for entry_type, content, thinking, tools_data in paired:
        if entry_type == "tool_call":
            for idx, (tool_name, args, output) in enumerate(tools_data):
                step_num += 1
                args_brief = json.dumps(args, ensure_ascii=False) if args else ""
                output_brief = output if output else ""
                step = {
                    "step": step_num,
                    "tool": tool_name,
                    "purpose": content if content else "",
                    "args_brief": args_brief,
                    "output_brief": output_brief,
                }
                # Attach thinking only to the first tool call of this AI message
                if idx == 0 and thinking:
                    step["thinking"] = thinking
                steps.append(step)
        else:
            step_num += 1
            step = {
                "step": step_num,
                "tool": None,
                "purpose": content,
            }
            if thinking:
                step["thinking"] = thinking
            steps.append(step)
    return steps


def _parse_from_folder(folder_path, embed_images=False, case_id="", model_key=""):
    """Parse trajectory from step files in a folder (has thinking field).
    If embed_images=True, save visual data to files and reference them.
    """
    step_files = sorted(glob.glob(os.path.join(folder_path, "step_*.json")),
                        key=lambda f: int(re.search(r'step_(\d+)', os.path.basename(f)).group(1)))
    steps = []
    step_num = 0

    # For image embedding: build frame_index → image_path lookup from frame_extraction outputs
    frame_lookup = {}  # frame_index → image_path (cumulative across all extractions)

    # Group: each AI file followed by its tool output files
    i = 0
    while i < len(step_files):
        sf = step_files[i]
        fname = os.path.basename(sf)
        with open(sf) as f:
            data = json.load(f)

        if data.get("role") == "ai":
            content = data.get("content", "") or ""
            thinking = data.get("thinking", "") or ""
            tool_calls = data.get("tool_calls") or []

            if tool_calls:
                # Collect subsequent tool output files
                j = i + 1
                for tc_idx, tc in enumerate(tool_calls):
                    tool_name = tc.get("name", "")
                    args = tc.get("args", {})
                    output_text = ""
                    tool_data_raw = None
                    if j < len(step_files):
                        with open(step_files[j]) as tf:
                            tool_data_raw = json.load(tf)
                        if tool_data_raw.get("role") == "tool":
                            output_text = _extract_tool_output_text(tool_data_raw.get("content", ""))
                            j += 1
                        else:
                            tool_data_raw = None

                    step_num += 1
                    args_brief = json.dumps(args, ensure_ascii=False) if args else ""
                    step = {
                        "step": step_num,
                        "tool": tool_name,
                        "purpose": content if content else "",
                        "args_brief": args_brief,
                        "output_brief": output_text,
                    }
                    if tc_idx == 0 and thinking:
                        step["thinking"] = thinking

                    # Embed images for visual tools in key cases
                    if embed_images and tool_data_raw and tool_name in VISUAL_TOOLS:
                        # Add input_args to tool_data for bbox frame matching (prefer file's own)
                        if "input_args" not in tool_data_raw:
                            tool_data_raw["input_args"] = args
                        imgs = extract_visual_data_for_step(
                            tool_data_raw, tool_name, case_id, model_key, step_num, frame_lookup)
                        if imgs:
                            step["images"] = imgs

                        # Update frame_lookup from frame_extraction outputs
                        if tool_name == "frame_extraction" and tool_data_raw:
                            _update_frame_lookup(frame_lookup, tool_data_raw)

                    steps.append(step)
                i = j
            else:
                step_num += 1
                step = {
                    "step": step_num,
                    "tool": None,
                    "purpose": content,
                }
                if thinking:
                    step["thinking"] = thinking
                steps.append(step)
                i += 1
        else:
            # Tool output without AI parent (shouldn't happen, skip)
            i += 1

    return steps


def _update_frame_lookup(frame_lookup, tool_data):
    """Update frame_index → image_path from a frame_extraction tool output."""
    content = tool_data.get("content", "")
    if isinstance(content, list):
        text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    else:
        text = str(content)
    try:
        data = json.loads(text) if text.strip().startswith("{") else json.loads(
            text.strip("` \n").lstrip("json\n"))
        for idx, fr in enumerate(data.get("frames", [])):
            path = fr.get("image_path", "")
            if os.path.exists(path):
                frame_lookup[idx] = path
    except Exception:
        pass


def compute_bench_stats(cases_data, baseline_records, gpt_records, claude_records, gemini_records, case_lookup):
    """Compute per-benchmark stats with final-round context from trajectory folders."""
    import glob as _glob

    case_qids = set(case_lookup.keys())

    # Video duration from baseline
    vid_dur_map = {}
    for r in baseline_records:
        short = strip_prefix(r["question_id"])
        vid_dur_map[short] = r.get("video_info", {}).get("duration", 0)

    def extract_text_len(content):
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(len(item.get("text", "")) if isinstance(item, dict) else len(str(item)) for item in content)
        return 0

    def analyze_traj_folder(folder_path):
        """Compute final-round context and tool I/O from trajectory step files.
        Final input = system + user + all AI content + all tool outputs - last AI output.
        (The last AI message is the OUTPUT of the final call, not input.)"""
        meta_path = os.path.join(folder_path, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            meta = json.load(f)

        sys_len = len(meta.get("system_prompt", ""))
        user_len = len(meta.get("user_message", ""))

        step_files = sorted(_glob.glob(os.path.join(folder_path, "step_*.json")))

        total_ai_chars = 0
        total_tool_input_chars = 0
        total_tool_output_chars = 0
        last_ai_output_chars = 0

        for sf in step_files:
            with open(sf) as f:
                step = json.load(f)
            role = step.get("role", "")
            if role == "ai":
                content = step.get("content", "") or ""
                total_ai_chars += len(content)
                for tc in (step.get("tool_calls") or []):
                    args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                    total_tool_input_chars += len(args_str)
                    total_ai_chars += len(args_str)  # args are part of AI message
                last_ai_output_chars = len(content)
            elif role == "tool":
                total_tool_output_chars += extract_text_len(step.get("content", ""))

        # Final round input = everything EXCEPT the last AI output
        final_input_chars = sys_len + user_len + total_ai_chars + total_tool_output_chars - last_ai_output_chars

        return {
            "final_input_chars": final_input_chars,
            "last_output_chars": last_ai_output_chars,
            "tool_input_chars": total_tool_input_chars,
            "tool_output_chars": total_tool_output_chars,
        }

    # Build traj folder lookups
    traj_lookups = {}
    for mk, dirname in [
        ("gpt-5.4", "s5_agent_results_tool_v12_gpt-5.4_trajectories"),
        ("claude-opus-4-6", "s5_agent_results_tool_v12_claude-opus-4-6_trajectories"),
    ]:
        td = os.path.join(DATA_DIR, dirname)
        lookup = {}
        if os.path.exists(td):
            for name in os.listdir(td):
                full = os.path.join(td, name)
                if os.path.isdir(full):
                    lookup[name] = full
        traj_lookups[mk] = lookup

    gem_lookup = {}
    for dirname in ["s5_agent_results_tool_v12_gemini-3-pro_trajectories",
                     "s5_agent_results_tool_v12_gemini-3.1-pro_trajectories"]:
        td = os.path.join(DATA_DIR, dirname)
        if os.path.exists(td):
            for name in os.listdir(td):
                full = os.path.join(td, name)
                if os.path.isdir(full):
                    gem_lookup[name] = full
    traj_lookups["gemini"] = gem_lookup

    bench_stats = {}
    for ds in ["lvbench", "longvideobench", "videomme"]:
        ds_stats = {}

        # Baseline: single call — separate image/text tokens
        bl_items = [r for r in baseline_records
                    if strip_prefix(r["question_id"]) in case_qids and r["dataset"] == ds]
        n = len(bl_items)
        if n:
            ds_stats["baseline"] = {
                "n": n,
                "avg_input_tokens": round(sum(r.get("token_usage", {}).get("input_tokens", 0) for r in bl_items) / n),
                "avg_image_tokens": round(sum(r.get("token_usage", {}).get("image_tokens", 0) for r in bl_items) / n),
                "avg_text_tokens": round(sum(r.get("token_usage", {}).get("text_tokens", 0) for r in bl_items) / n),
                "avg_output_tokens": round(sum(r.get("token_usage", {}).get("output_tokens", 0) for r in bl_items) / n),
                "accuracy": round(sum(1 for r in bl_items if r.get("correct")) / n * 100, 1),
                "avg_vid_duration": round(sum(r.get("video_info", {}).get("duration", 0) for r in bl_items) / n, 1),
            }

        # Tool models: compute final-round context from trajectory folders
        for mk, recs in [("gpt-5.4", gpt_records), ("claude-opus-4-6", claude_records), ("gemini", gemini_records)]:
            items = [r for r in recs
                     if strip_prefix(r["question_id"]) in case_qids
                     and r.get("_meta") is None
                     and r["dataset"] == ds]
            n = len(items)
            if not n:
                continue

            traj_results = []
            for r in items:
                folder = traj_lookups.get(mk, {}).get(r["question_id"])
                if folder:
                    result = analyze_traj_folder(folder)
                    if result:
                        traj_results.append(result)

            nt = len(traj_results)
            ds_stats[mk] = {
                "n": n,
                "avg_final_input_chars": round(sum(t["final_input_chars"] for t in traj_results) / max(nt, 1)) if nt else 0,
                "avg_last_output_chars": round(sum(t["last_output_chars"] for t in traj_results) / max(nt, 1)) if nt else 0,
                "avg_tool_input_chars": round(sum(t["tool_input_chars"] for t in traj_results) / max(nt, 1)) if nt else 0,
                "avg_tool_output_chars": round(sum(t["tool_output_chars"] for t in traj_results) / max(nt, 1)) if nt else 0,
                "avg_steps": round(sum(r.get("steps", 0) for r in items) / n, 1),
                "accuracy": round(sum(1 for r in items if r.get("correct")) / n * 100, 1),
                "avg_vid_duration": round(sum(vid_dur_map.get(strip_prefix(r["question_id"]), 0) for r in items) / n, 1),
                "traj_coverage": nt,
            }

        bench_stats[ds] = ds_stats

    return bench_stats


# ── Annotations for key cases ──
# Each case has: summary (overall analysis), baseline_note, and per-model step annotations
# Step annotations: {model_key: {step_num: "annotation text"}}
CASE_ANNOTATIONS = {
    # ══════════ Cat1: BL✓ All-Tool✗ ══════════
    "hf4WUOagFAw_1": {
        "category": "cat1",
        "tag": "Baseline✓ All-Tool✗",
        "summary": "K-pop 舞台表演视频（216s），需要在视频末尾找到穿紫色上衣比 7 手势闭眼的人，再回溯其第一次出现时的动作。三个工具模型全部错选 E（拿麦跳舞），GT 是 B（坐在舞台上伸展）。",
        "failure_reason": "多步时间回溯中人物跟踪丢失：从 211s 回溯到 13s 需要跨越大量帧，工具的 temporal_grounding 对'第一次出现'的定位不稳定（有的返回 13s、有的 20s、有的 37s），最终都锁定到了该人拿麦跳舞的片段而非最早的坐姿片段。",
        "baseline_note": "Baseline 直接观看完整视频帧，一次性看到 13s 时该人坐在舞台地板上的画面，与 211s 的紫色上衣形成正确关联。端到端视觉理解在人物跨时间匹配方面有天然优势。",
        "model_notes": {
            "gpt-5.4": "24 步中花费大量步骤做 frame_extraction + object_detection，最终定位到 36-40s 该人拿麦跳舞的片段，错过了更早的 13s 坐姿。",
            "claude-opus-4-6": "72 步（最多），第 35 步实际检测到 13s 坐姿+手靠嘴，第 39 步检测到 leaning back，但在最终推理时被大量矛盾证据淹没，犹豫后选了 E。过度调用工具反而引入噪声。",
            "gemini": "17 步，将第一次出现定位到 ~20s 拿麦唱歌，完全跳过了 13s 的坐姿。temporal_grounding 对'first appearance'理解有偏差。",
        },
        "step_annotations": {
            "gpt-5.4": {
                2: "frame_extraction 采样了 12 帧全局概览，但分辨率不足以识别紫色上衣人物",
                9: "object_detection 在后半段密集检测近景镜头中的人物",
                16: "frame_extraction 全局重新搜索 purple top，前面都是黑色衣服",
                19: "temporal_grounding 定位到 211-212s 最后出现，但回溯首次出现时定位到 37-40s",
                22: "action_recognition 在 36-40s 识别到 dancing with microphone → 错误锚定",
                24: "object_reidentification 尝试跨时间匹配但失败。最终基于错误定位选了 E",
            },
            "claude-opus-4-6": {
                3: "temporal_grounding 找到 210s 紫色上衣 + 七手势",
                17: "attribute_recognition 确认 13.5s purple_top=yes, action=dancing",
                34: "attribute_recognition 确认紫色在 frame_index 6（13.5s 右下角）",
                39: "attribute_recognition 关键发现：13.5s sitting on floor, hand near mouth——正确答案的直接证据！",
                62: "temporal_grounding 再次确认 13s sitting + 13.5s holding microphone",
                71: "object_tracking 尝试跟踪紫色人物移动。最终推理列出 B 选项证据，但被大量 dancing 证据干扰选了 E",
            },
            "gemini": {
                2: "frame_extraction 全局采样 15 帧",
                11: "frame_extraction 提取视频开头 0-20s 的帧，查找首次出现",
                16: "attribute_recognition 检查 10s 和 20s 帧，判断首次出现在 20s 拿麦唱歌",
                17: "attribute_recognition 验证指向动作，基于错误的首次出现定位选了 E",
            },
        },
    },

    # ══════════ Cat3: BL✗ All-Tool✓ ══════════
    "1069": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "长视频（2665s）中辨认戴蓝色短袖+眼镜的人佩戴的手表颜色。GT=B 黑色。Baseline 误判为金色（A），三个工具模型均正确。",
        "failure_reason": "Baseline 在 2665s 长视频中帧采样密度不足，手表细节在缩略帧中难以辨认，误将金色表扣/棕色表带判断为金色手表。",
        "baseline_note": "Baseline 回答 A（Golden），从 response 看它注意到了手表有 'gold-colored face and brown leather strap'，但实际手表表盘是黑色的，表扣可能是金色。长视频低分辨率帧下的细节判断出错。",
        "model_notes": {
            "gpt-5.4": "26 步，通过反复 object_detection + spatial_crop 定位手腕区域并放大，最终识别出黑色表盘。",
            "claude-opus-4-6": "仅 8 步，高效定位 → crop → 识别，快速得出正确答案。",
            "gemini": "23 步，也通过 crop + attribute_recognition 确认黑色手表。",
        },
        "step_annotations": {
            "gpt-5.4": {
                2: "frame_extraction 在 2665s 长视频中采样关键帧",
                4: "object_detection 开始在各帧中搜索手表（连续多步检测）",
                16: "frame_extraction 聚焦到候选帧段",
                23: "spatial_crop 放大手腕区域——工具方法的核心优势，近距离看清手表细节",
                26: "attribute_recognition 确认黑色表盘",
            },
            "claude-opus-4-6": {
                2: "temporal_grounding 快速定位蓝色短袖+眼镜出现的时段",
                3: "frame_extraction 提取候选帧",
                7: "attribute_recognition 确认手表存在",
                8: "spatial_crop 裁剪手腕放大——高效地 8 步完成任务",
            },
        },
    },
    "1135": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "长视频（2960s）中判断 06:01 时一个男人的表情。GT=D 笑。Baseline 误判为麻木（A），工具模型均正确。",
        "failure_reason": "Baseline 帧采样间隔太大，可能未采样到 06:01 附近的帧，或在缩略帧中表情细节丢失。",
        "baseline_note": "Baseline 判断 'neutral and serious, not crying/angry/laughing → numb'。但 06:01 处该男子实际在笑。帧采样粒度不够精确。",
        "model_notes": {
            "gpt-5.4": "仅 4 步！精确提取 06:01 附近帧，直接识别出笑容。",
            "claude-opus-4-6": "3 步，同样快速定位 + 识别。",
            "gemini": "9 步，稍多但同样正确。",
        },
        "step_annotations": {
            "gpt-5.4": {
                2: "frame_extraction 精确提取 361s（06:01）附近的帧",
                4: "facial_emotion_recognition 直接识别出 happy——工具的时间定位+表情识别精度是关键优势",
            },
            "claude-opus-4-6": {
                2: "frame_extraction 精确定位 06:01",
                3: "facial_emotion_recognition 检测到 happy，仅 3 步完成",
            },
        },
    },
    "2362": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "超长视频（7547s/2h+）中定位小美人鱼第一次出现的时间。GT=B（03:40）。Baseline 选了 D（02:40），工具模型全对。",
        "failure_reason": "7547s 超长视频，Baseline 帧采样覆盖不足，在时间定位精度上天然劣势。",
        "baseline_note": "Baseline 在 2 小时视频中逐帧扫描，但采样间隔过大导致定位不准，误判为 02:40。",
        "model_notes": {
            "gpt-5.4": "8 步，使用 temporal_grounding 精准搜索 'little mermaid' 首次出现。",
            "claude-opus-4-6": "10 步，同样通过 temporal_grounding 定位。",
            "gemini": "20 步，多次验证确认时间点。",
        },
        "step_annotations": {
            "gpt-5.4": {
                6: "temporal_grounding 搜索 'little mermaid' 首次出现——语义搜索在 7547s 超长视频中的杀手锏",
                8: "frame_comparison 确认 03:40 首次出现角色",
            },
            "claude-opus-4-6": {
                2: "temporal_grounding 在 2h+ 视频中搜索小美人鱼出现时段",
                9: "frame_extraction 精细提取 220-240s 帧定位精确时间",
                10: "frame_comparison 确认首次出现时间",
            },
        },
    },
    "4023": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "网球比赛视频（2792s），问在特定比分时镜头对准数字的原因。GT=A（发球速度极高）。Baseline 选 C（球击中了数字），工具模型全对。",
        "failure_reason": "Baseline 无法精确读取速度数字，误判了因果关系。",
        "baseline_note": "Baseline 看到镜头对准数字但无法用 OCR 精确读取速度值，误以为是球击中了显示屏上的数字。",
        "model_notes": {
            "gpt-5.4": "13 步，通过 OCR + 帧提取读出速度数字为极高值。",
            "claude-opus-4-6": "23 步，用 text_recognition 精确读出发球速度。",
            "gemini": "15 步，同样通过 OCR 读数。",
        },
        "step_annotations": {
            "gpt-5.4": {
                4: "text_recognition OCR 读取比分和速度信息",
                9: "text_recognition 在候选帧上精确 OCR",
                12: "text_recognition 精确读出速度数字 192——OCR 工具的精确读数能力 Baseline 做不到",
                13: "frame_comparison 确认大号 '192' 出现在画面中，对应发球速度极高",
            },
            "claude-opus-4-6": {
                2: "temporal_grounding 搜索 'serve speed number'",
                6: "text_recognition OCR 读取比分板",
                18: "frame_extraction 精细定位 15-15 比分时刻",
                22: "spatial_crop 裁剪速度显示区域，确认 192 km/h",
            },
        },
    },
    "4754": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "长视频（2450s）中辨认 20:29 画面最近的人穿什么衣服。GT=C（条纹衫）。Baseline 选 D（格子衫），工具模型全对。",
        "failure_reason": "Baseline 在低分辨率帧中将条纹误判为格子。细节纹理在缩略帧中容易混淆。",
        "baseline_note": "Baseline 简短回答 'plaid shirt'（格子衫），可能在缩小帧中条纹和格子的纹理难以区分。",
        "model_notes": {
            "gpt-5.4": "11 步，通过 spatial_crop 放大人物服装区域，清晰看到条纹纹理。",
            "claude-opus-4-6": "17 步，crop + attribute_recognition 确认条纹。",
            "gemini": "9 步，高效完成。",
        },
        "step_annotations": {
            "gpt-5.4": {
                4: "object_detection 定位 20:29 画面中所有人物",
                5: "attribute_recognition 初步识别出 striped shirt",
                6: "spatial_crop 裁剪每个人物区域放大——条纹 vs 格子在放大后一目了然",
                9: "attribute_recognition 对放大后的 crop 逐一确认衣服纹理",
            },
            "claude-opus-4-6": {
                3: "object_detection 定位人物 + 画框位置",
                4: "spatial_crop 裁剪放大候选人物",
                9: "spatial_analysis 计算哪个人离画最近",
                14: "attribute_recognition 确认 'striped shirt'",
            },
        },
    },
    "4QSmRYQBfN4_0": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "短视频（17s）中识别旗帜图案细节：字幕后出现了什么。GT=C（白色箭头 + 红旗上两个黄色图案）。Baseline 选 D，工具模型全对。",
        "failure_reason": "Baseline 在快速切换的画面中遗漏了部分视觉元素的共现。",
        "baseline_note": "虽然只有 17s，但画面切换快且细节密集。Baseline 正确识别了箭头和旗帜，但对旗帜上的黄色图案数量判断错误。",
        "model_notes": {
            "gpt-5.4": "10 步，通过密集帧提取捕捉快速切换的画面。",
            "claude-opus-4-6": "115 步（极多！），对这个 17s 视频做了极其详尽的逐帧分析。",
            "gemini": "18 步，通过 spatial_crop 放大旗帜细节。",
        },
        "step_annotations": {
            "gpt-5.4": {
                2: "frame_extraction 在 17s 视频中密集提取帧，捕获所有关键画面",
                3: "text_recognition OCR 读取字幕文本，定位 'machete with half a cogwheel'",
                5: "frame_comparison 对比关键帧，发现白色箭头+旗帜变化",
                8: "spatial_crop 裁剪右下角区域放大分析",
            },
        },
    },
    "011-3": {
        "category": "cat3",
        "tag": "Baseline✗ All-Tool✓",
        "summary": "芭蕾舞视频（118s），识别最后一幕的舞蹈动作。GT=A（单膝跪地后仰）。Baseline 选 D（passé + Grand jeté），工具模型全对。",
        "failure_reason": "Baseline 对专业芭蕾术语的视觉对应理解不准确。",
        "baseline_note": "Baseline 在最后几帧中看到了抬腿和弯膝动作，误判为 passé + Grand jeté。实际上最终动作是单膝跪地后仰。",
        "model_notes": {
            "gpt-5.4": "9 步，提取最后几秒帧 + pose_estimation 确认跪地后仰。",
            "claude-opus-4-6": "5 步，快速高效。",
            "gemini": "4 步，最精简。",
        },
        "step_annotations": {
            "gpt-5.4": {
                3: "shot_boundary_detection 检测到最后一个镜头在 102.3s",
                4: "action_recognition 初步分析最后场景动作",
                5: "frame_extraction 密集提取 102-113s 的帧",
                8: "action_recognition 识别 'kneel down and lean back'——精确匹配选项 A",
                9: "pose_estimation 确认单膝跪地后仰姿势",
            },
            "claude-opus-4-6": {
                2: "shot_boundary_detection 快速定位最后场景",
                4: "action_recognition 分析舞蹈动作",
                5: "pose_estimation 确认跪地后仰——5 步完成",
            },
            "gemini": {
                3: "shot_boundary_detection 定位最终场景",
                4: "action_recognition 识别 kneel + lean back——仅 4 步工具调用",
            },
        },
    },

    # ══════════ Cat2: BL✓ + 2 Tools✗ ══════════
    "053-1": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "岩浆颜色变化视频（93s），问岩浆暴露在空气中短时间后颜色如何变化。GT=C（变银色）。GPT 选 B（红色），Claude 选 D（黑色），Gemini 正确。",
        "failure_reason": "岩浆冷却过程涉及多阶段颜色变化：红/橙 → 银灰色 → 黑色。GPT 和 Claude 分别锁定了不同的错误阶段。关键词是 'short while'（短时间），对应银灰色过渡态。",
        "baseline_note": "Baseline 直接观察到岩浆从明亮橙红色到冷却表面的颜色变化，正确识别出短时间暴露后的银灰色外壳。端到端视频理解在连续颜色渐变观察上有优势。",
        "model_notes": {
            "gpt-5.4": "11 步，尝试通过 speech_transcription 和 text_recognition 获取解说但未找到明确描述，最终 frame_comparison 只观察到红色和黑色两个极端状态，选了 B（红色）。",
            "claude-opus-4-6": "10 步，attribute_recognition 检测到颜色从 orange → red → black 的变化，frame_comparison 提到了 'silvery-grey' 但最终推理时未能将其与 'short while' 关联，选了 D（黑色）。",
            "gemini": "仅 4 步，高效完成。通过 frame_extraction + audio 分析准确判断银色。",
        },
        "step_annotations": {
            "gpt-5.4": {
                3: "frame_extraction 采样关键帧，但帧间隔可能跳过了银灰色过渡态",
                10: "frame_comparison 只对比了红色熔岩和黑色冷却面，遗漏了中间的银灰色阶段",
            },
            "claude-opus-4-6": {
                3: "attribute_recognition 检测到颜色序列 orange → red → black",
                8: "frame_comparison 提到了 'metallic silvery-grey' 冷却外壳——正确证据出现但被忽略",
                9: "text_recognition 未找到字幕解说。最终选了黑色而非银色",
            },
        },
    },
    "416-1": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "企鹅视频（260s），问视频开头小企鹅的移动方向。GT=B（从左到右）。Claude 选 C（从右到左），Gemini 选 A（原地不动），GPT 正确。",
        "failure_reason": "小企鹅在开头几秒移动幅度很小，工具的 object_tracking 和 bbox 变化检测精度不足，Claude 误判方向，Gemini 判断为静止。",
        "baseline_note": "Baseline 直接看视频帧就能判断企鹅从左向右移动，简单直观的视觉判断不需要工具辅助。",
        "model_notes": {
            "gpt-5.4": "6 步，frame_comparison + object_detection 检测 bbox 位移，正确判断从左到右。",
            "claude-opus-4-6": "10 步，object_tracking 报告企鹅 'stationary'，但 bbox 数据有噪声：x 坐标在 519-723 间振荡。Claude 取首尾差值判断为从右到左，实际是噪声干扰。",
            "gemini": "9 步，action_recognition 检测到 'standing'，object_tracking 判断位置不变，得出原地不动。工具对微小移动的检测灵敏度不够。",
        },
        "step_annotations": {
            "claude-opus-4-6": {
                3: "object_tracking 报告企鹅 'stationary'，与后续 bbox 分析矛盾",
                7: "object_detection 获取 bbox 但 x 坐标在帧间有较大噪声波动",
                9: "camera_motion_analysis 确认镜头静止，但 bbox 噪声导致方向判断反转",
            },
            "gemini": {
                6: "object_tracking 判断企鹅 'stationary, shivering'，遗漏了缓慢的左→右位移",
                8: "action_recognition 检测到 'standing' 而非 'walking'，确认了静止的错误判断",
            },
        },
    },
    "63": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "历史日剧（3666s），问主角在香炉里插了几根香。GT=D（1根）。GPT 和 Claude 都选 A（3根），Gemini 正确。",
        "failure_reason": "香炉中最终可见 3 根香（含之前已存在的），但主角本次只插了 1 根。工具检测到的是最终状态的数量，而非动作增量。",
        "baseline_note": "Baseline 直接观看完整动作序列，正确识别出主角拿着一根香点燃并插入，计数准确。",
        "model_notes": {
            "gpt-5.4": "12 步，temporal_grounding 定位到香炉场景，但 frame_comparison 看到最终 4 根香（含已有的），推断插入了 3 根。混淆了'已有数量'和'插入数量'。",
            "claude-opus-4-6": "33 步（过度分析），多次 spatial_crop 放大香炉区域，attribute_recognition 检测到'1 stick being held'，但后续看到最终 3 根就改判为 A。前后证据矛盾时选了错误的。",
            "gemini": "18 步，通过密集 frame_comparison 跟踪动作序列，正确识别出只插入了 1 根。",
        },
        "step_annotations": {
            "gpt-5.4": {
                5: "temporal_grounding 定位到 682-694s 的香炉场景",
                11: "frame_comparison 看到最终 4 根香，错误推断插入了 3 根。应区分已有 vs 新增",
            },
            "claude-opus-4-6": {
                2: "temporal_grounding 搜索 'incense burner' 场景",
                8: "spatial_crop 放大香炉区域观察细节",
                21: "attribute_recognition 检测到 '1 stick being held'——这是正确答案的直接证据！",
                31: "spatial_crop 看到最终 3 根香后改判 A。被'最终状态'误导，忽略了之前 step 21 的证据",
            },
        },
    },
    "135": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "长视频（3296s），问 37:30-38:05 发生了什么。GT=A（男子重访两人去过的地方）。GPT 选 B（新地方），Claude 选 C（去过+想去+新地方），Gemini 正确。",
        "failure_reason": "需要对比 37:30 片段中的场景与视频前半段出现过的场景，判断是'重访'还是'新地方'。工具无法进行跨时段场景匹配。",
        "baseline_note": "Baseline 直接观看完整视频帧，能将 37:30 的蒙太奇场景与前面出现过的场景对应起来，判断为重访旧地。",
        "model_notes": {
            "gpt-5.4": "13 步，scene_recognition 和 frame_comparison 只识别出'多个户外场景的蒙太奇'，但无法判断这些场景是否之前出现过。选了 B（新地方）。",
            "claude-opus-4-6": "31 步，大量 temporal_grounding 尝试搜索之前出现过的场景，但搜索结果都返回高置信度匹配，导致误判为'去过+想去+新地方都有'。",
            "gemini": "10 步，高效正确判断。",
        },
        "step_annotations": {
            "gpt-5.4": {
                3: "scene_recognition 识别出雨天户外蒙太奇，但无法判断是否为旧地",
                7: "frame_comparison 对比帧间变化，只看到场景切换但无法与前半段匹配",
                12: "frame_comparison 最终只能判断'多个不同地点'，缺乏跨时段对比能力",
            },
            "claude-opus-4-6": {
                9: "temporal_grounding 搜索'man visits places he and woman have been to'，返回高置信度",
                20: "temporal_grounding 尝试搜索'new places'和'would like to go'，也返回高置信度——搜索结果不可靠",
                23: "frame_comparison 发现动画角色信息，但仍无法区分'旧地'和'新地'",
            },
        },
    },
    "019-2": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "乌鸦视频（107s），问乌鸦在做什么。GT=A（吃东西）。GPT 选 B（飞），Claude 选 C（走），Gemini 正确。",
        "failure_reason": "乌鸦在地面啄食，action_recognition 对鸟类的细粒度动作识别不佳，将'啄食'误判为'飞'或'走'。",
        "baseline_note": "Baseline 直接看到乌鸦低头啄地的画面，简单直观地判断为吃东西。视觉常识推理不需要工具。",
        "model_notes": {
            "gpt-5.4": "10 步，action_recognition 返回 'flying' 作为主要动作。后续 frame_comparison 也未能纠正，选了 B。",
            "claude-opus-4-6": "22 步，大量 object_detection 和 temporal_grounding，scene_recognition 识别出'黑色鸟在草地上'。但 action_recognition 返回 walking，最终选了 C。",
            "gemini": "仅 3 步，frame_extraction + object_detection 直接判断出吃东西。",
        },
        "step_annotations": {
            "gpt-5.4": {
                3: "action_recognition 返回 'flying'——对鸟类地面啄食动作的误判",
                9: "frame_comparison 观察到乌鸦在地面但未纠正 action_recognition 的错误判断",
            },
            "claude-opus-4-6": {
                7: "action_recognition 返回 'walking'——与 GPT 错误方向不同但同样错误",
                9: "temporal_grounding 搜索 'raven eating'，但结果不够明确",
                21: "attribute_recognition 分析乌鸦姿态，最终仍选了 walking",
            },
        },
    },
    "619-2": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "陶瓷制作视频（1980s），问烧制前陶瓷的颜色。GT=C（棕色）。GPT 和 Claude 都选 B（白色），Gemini 正确。",
        "failure_reason": "陶瓷制作有多个阶段：棕色泥坯 → 成型后表面变浅（白色/米色）→ 上釉前的状态。工具提取的帧可能是上釉或半干状态，看起来偏白/浅色。",
        "baseline_note": "Baseline 观察到完整的制作过程，从始至终看到泥坯是棕色的。正确判断烧制前为棕色。",
        "model_notes": {
            "gpt-5.4": "14 步，attribute_recognition 检测到 'white/tan/light brown'。在白色和棕色之间犹豫，最终选了白色。",
            "claude-opus-4-6": "18 步，attribute_recognition 检测到 'off-white'，虽然在 180s 处发现了 brown 的 raw 状态，但最终被后续帧的 white 证据覆盖。",
            "gemini": "12 步，正确判断棕色。",
        },
        "step_annotations": {
            "gpt-5.4": {
                7: "temporal_grounding 搜索 ceramics before furnace 场景",
                9: "attribute_recognition 检测到 'white/tan/light brown'——颜色判断模糊",
            },
            "claude-opus-4-6": {
                4: "attribute_recognition 检测到 'off-white' 颜色",
                10: "attribute_recognition 在视频开头发现 'brown' raw 状态——正确证据出现但被后续覆盖",
                17: "attribute_recognition 最终确认 'white'。被半成品/干燥后的浅色误导",
            },
        },
    },
    "61": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "历史日剧（3666s），持枪人威胁厨师后主角做什么。GT=A（推桌站起对峙→杀人→离开→厨师跟随）。Claude 选 B（injuries 而非 kills），Gemini 选 D（无厨师跟随）。",
        "failure_reason": "选项 A 和 B 的区别仅在于 kills vs injuries；A 和 D 的区别在于厨师是否跟随。需要精确判断这些细节。",
        "baseline_note": "Baseline 看完整个场景序列，正确识别出主角杀了持枪人且厨师跟随离开。",
        "model_notes": {
            "gpt-5.4": "17 步，通过密集分析正确判断：无枪、杀人、厨师跟随。选 A 正确。",
            "claude-opus-4-6": "28 步，正确判断无枪和厨师跟随，但在 kills vs injuries 上判断错误。frame_comparison 显示的动作模糊，无法确定结果是死亡还是受伤。选了 B。",
            "gemini": "17 步，正确判断杀人，但遗漏了厨师跟随的细节。选了 D。",
        },
        "step_annotations": {
            "claude-opus-4-6": {
                7: "frame_extraction 提取对峙场景的关键帧",
                22: "frame_comparison 分析战斗结果，但无法确定是 kills 还是 injuries",
                25: "frame_comparison 观察到对方倒地但判断为受伤而非死亡。选了 B",
            },
            "gemini": {
                5: "action_recognition 分析战斗动作",
                11: "frame_extraction 提取离开场景",
                16: "action_recognition 分析离开后的动作，但遗漏了厨师跟随的细节",
            },
        },
    },
    "41": {
        "category": "cat2",
        "tag": "Baseline✓ 2-Tool✗",
        "summary": "长视频（2044s），问主角在 10:33 如何通过通道。GT=B（跑过但腿受伤）。GPT 答案提取失败（None），Gemini 选 C（丢失武器）。Claude 正确。",
        "failure_reason": "GPT 仅 2 步就因 frame_extraction 失败而放弃。Gemini 在密集动作场景中将长矛刺伤误判为丢失武器。",
        "baseline_note": "Baseline 看到主角跑过通道时被长矛刺中腿部，然后拔出继续前进。正确判断为腿受伤。",
        "model_notes": {
            "gpt-5.4": "仅 2 步！frame_extraction 后直接结束，答案提取失败返回 None。这是工具方法最严重的失败模式——流程中断。",
            "claude-opus-4-6": "19 步，精确定位到 633-676s 的通道场景，通过 temporal_grounding 找到腿部受伤片段。选 B 正确。",
            "gemini": "17 步，action_recognition 检测到 'running and dodging'，但将矛刺伤误解为丢失武器。选了 C。",
        },
        "step_annotations": {
            "gpt-5.4": {
                2: "frame_extraction 提取 10:33 附近帧后直接结束，未进行任何分析。答案为 None",
            },
            "gemini": {
                3: "action_recognition 检测到 running and dodging",
                9: "frame_comparison 分析动作细节，但将矛刺伤误读为武器脱手",
                15: "action_recognition 继续分析但未能纠正错误判断",
            },
        },
    },
}


def main():
    # ── Clean up old generated images ──
    import shutil
    if os.path.exists(IMG_OUTPUT_DIR):
        shutil.rmtree(IMG_OUTPUT_DIR)
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
    _img_counter[0] = 0

    # ── Load data ──
    reg_data = load_json(os.path.join(DATA_DIR, "s4_tool_registry_v12.json"))
    cases_data = load_json(os.path.join(DATA_DIR, "s2_case_abilities_116.json"))
    gemini_records = load_jsonl(os.path.join(DATA_DIR, "121case-gemini_3.1or3.0-tool_v10orv12.jsonl"))
    gpt_records = load_jsonl(os.path.join(DATA_DIR, "s5_agent_results_tool_v12_gpt-5.4.jsonl"))
    claude_records = load_jsonl(os.path.join(DATA_DIR, "s5_agent_results_tool_v12_claude-opus-4-6.jsonl"))
    baseline_records = load_jsonl(os.path.join(DATA_DIR, "s5_agent_results_direct_gemini-3.1-pro.jsonl"))
    comp_data = load_json(os.path.join(VIEWER_DIR, "data/execution-comparison-v2.json"))

    # ── Case lookup ──
    case_lookup = {}
    for bench, qlist in cases_data["cases_by_benchmark"].items():
        for q in qlist:
            case_lookup[q["question_id"]] = {
                "question_id": q["question_id"],
                "ability_ids": q["ability_ids"],
                "ability_names": q["ability_names"],
                "question_text": q["question_text"],
                "benchmark": bench,
            }

    # ── Build per-model trajectory lookups (short qid → record) ──
    def build_lookup(records):
        lookup = {}
        for rec in records:
            short = strip_prefix(rec["question_id"])
            lookup[short] = rec
        return lookup

    gemini_lookup = build_lookup(gemini_records)
    gpt_lookup = build_lookup(gpt_records)
    claude_lookup = build_lookup(claude_records)
    baseline_lookup = build_lookup(baseline_records)

    # ── Fix question_text: use full question + options from JSONL (s2 file truncates at 200 chars) ──
    for short_qid, case in case_lookup.items():
        for lookup in [gpt_lookup, claude_lookup, gemini_lookup, baseline_lookup]:
            rec = lookup.get(short_qid)
            if rec and rec.get("question"):
                full_text = rec["question"]
                opts = rec.get("options", [])
                if opts:
                    full_text += "\n" + "\n".join(opts)
                case["question_text"] = full_text
                break

    # ── YouTube map ──
    yt_map = {}
    for q in comp_data.get("questions", []):
        url = q.get("youtube_url", "")
        if url:
            yt_map[strip_prefix(q.get("question_id", ""))] = url

    # ── Video duration map from baseline ──
    vid_dur_map = {}
    for r in baseline_records:
        short = strip_prefix(r["question_id"])
        vid_dur_map[short] = r.get("video_info", {}).get("duration", 0)

    # ── Abilities ──
    abilities = []
    for a in cases_data["abilities"]:
        abilities.append({
            "ability_id": a["id"],
            "ability_name": a["name"],
            "ability_category": ABILITY_CATEGORY[a["id"]],
            "ability_description": a["name"],
        })

    # ── Build trajectory folder lookups (for thinking field) ──
    traj_folder_lookups = {}
    for mk, dirname in [
        ("gpt-5.4", "s5_agent_results_tool_v12_gpt-5.4_trajectories"),
        ("claude-opus-4-6", "s5_agent_results_tool_v12_claude-opus-4-6_trajectories"),
    ]:
        td = os.path.join(DATA_DIR, dirname)
        lookup = {}
        if os.path.exists(td):
            for name in os.listdir(td):
                full = os.path.join(td, name)
                if os.path.isdir(full):
                    lookup[name] = full
        traj_folder_lookups[mk] = lookup

    gem_folder_lookup = {}
    for dirname in ["s5_agent_results_tool_v12_gemini-3-pro_trajectories",
                     "s5_agent_results_tool_v12_gemini-3.1-pro_trajectories"]:
        td = os.path.join(DATA_DIR, dirname)
        if os.path.exists(td):
            for name in os.listdir(td):
                full = os.path.join(td, name)
                if os.path.isdir(full):
                    gem_folder_lookup[name] = full
    traj_folder_lookups["gemini"] = gem_folder_lookup

    # ── Build questions with 3-model trajectories ──
    BENCH_ORDER = {"LVBench": 0, "LongVideoBench": 1, "Video-MME": 2}
    # Canonical model keys
    MODELS = ["gpt-5.4", "claude-opus-4-6", "gemini"]

    questions = []
    for short_qid, case in case_lookup.items():
        aid_str = case["ability_ids"]
        primary_aid = int(aid_str.split(",")[0])

        trajectories = {}
        per_model_info = {}

        for model_key, lookup in [("gpt-5.4", gpt_lookup), ("claude-opus-4-6", claude_lookup), ("gemini", gemini_lookup)]:
            rec = lookup.get(short_qid)
            if rec:
                # Use trajectory folder if available (has thinking field)
                traj_folder = traj_folder_lookups.get(model_key, {}).get(rec["question_id"])
                is_key_case = rec["question_id"] in KEY_VISUAL_CASES
                steps = parse_trajectory_steps(rec.get("trajectory", []), traj_folder=traj_folder,
                                                embed_images=is_key_case,
                                                case_id=rec["question_id"], model_key=model_key)
                trajectories[model_key] = {"steps": steps}
                per_model_info[model_key] = {
                    "correct": rec.get("correct", False),
                    "agent_answer": rec.get("agent_answer", ""),
                    "steps_count": rec.get("steps", 0),
                    "duration_seconds": rec.get("duration_seconds", 0),
                    "model_id": rec.get("model", model_key),
                    "source_model": rec.get("source_model", ""),
                    "source_tool_version": rec.get("source_tool_version", "v12"),
                }

        if not trajectories:
            continue

        # Baseline data
        baseline_rec = baseline_lookup.get(short_qid)
        baseline = None
        if baseline_rec:
            response_text = baseline_rec.get("response", "") or ""
            reasoning_text = baseline_rec.get("reasoning_content", "") or ""
            baseline = {
                "correct": baseline_rec.get("correct", False),
                "answer": baseline_rec.get("agent_answer", ""),
                "response": response_text,
                "reasoning": reasoning_text,
            }

        # Annotations for key cases
        ann = CASE_ANNOTATIONS.get(short_qid)
        ann_data = None
        if ann:
            ann_data = {
                "category": ann["category"],
                "tag": ann["tag"],
                "summary": ann["summary"],
                "failure_reason": ann.get("failure_reason", ""),
                "baseline_note": ann.get("baseline_note", ""),
                "model_notes": ann.get("model_notes", {}),
                "step_annotations": ann.get("step_annotations", {}),
            }

        questions.append({
            "question_id": short_qid,
            "ability_id": primary_aid,
            "question_text": case["question_text"],
            "benchmark": case["benchmark"],
            "youtube_url": yt_map.get(short_qid, ""),
            "video_duration": round(vid_dur_map.get(short_qid, 0), 1),
            "ground_truth": per_model_info.get("gpt-5.4", per_model_info.get("gemini", {})).get("agent_answer", "?"),  # GT from any
            "trajectories": trajectories,
            "per_model": per_model_info,
            "baseline": baseline,
            "annotations": ann_data,
        })

    # Fix ground_truth — get from actual record
    for q in questions:
        for mk, lookup in [("gpt-5.4", gpt_lookup), ("claude-opus-4-6", claude_lookup), ("gemini", gemini_lookup)]:
            rec = lookup.get(q["question_id"])
            if rec and rec.get("ground_truth"):
                q["ground_truth"] = rec["ground_truth"]
                break

    # Sort and number
    questions.sort(key=lambda q: (BENCH_ORDER.get(q["benchmark"], 99), q["ability_id"], q["question_id"]))
    for i, q in enumerate(questions):
        q["sequential_id"] = i + 1

    # ── Tool usage stats (across all models) ──
    tool_usage = defaultdict(lambda: {"frequency": 0, "ability_ids": set()})
    per_tool_per_model = defaultdict(lambda: defaultdict(int))
    per_tool_total = defaultdict(int)

    model_stats_raw = {m: {"total_steps": 0, "tools_used": set(), "trajectories": 0, "correct": 0} for m in MODELS}

    for q in questions:
        for mk in MODELS:
            traj = q["trajectories"].get(mk)
            info = q["per_model"].get(mk)
            if not traj or not info:
                continue
            model_stats_raw[mk]["trajectories"] += 1
            model_stats_raw[mk]["total_steps"] += info["steps_count"]
            if info["correct"]:
                model_stats_raw[mk]["correct"] += 1
            for s in traj["steps"]:
                tname = s.get("tool")
                if tname:
                    tool_usage[tname]["frequency"] += 1
                    tool_usage[tname]["ability_ids"].add(q["ability_id"])
                    per_tool_per_model[tname][mk] += 1
                    per_tool_total[tname] += 1
                    model_stats_raw[mk]["tools_used"].add(tname)

    # Finalize model stats
    model_stats = {}
    for m in MODELS:
        raw = model_stats_raw[m]
        n = max(raw["trajectories"], 1)
        model_stats[m] = {
            "avg_steps": round(raw["total_steps"] / n, 1),
            "total_steps": raw["total_steps"],
            "total_tools_used": len(raw["tools_used"]),
            "trajectories": raw["trajectories"],
            "correct": raw["correct"],
            "accuracy": round(raw["correct"] / n * 100, 1),
        }

    traj_stats = {
        "models": model_stats,
        "per_tool": {k: dict(v) for k, v in per_tool_per_model.items()},
        "per_tool_total": dict(per_tool_total),
    }

    # ── Build REGISTRY ──
    reg_tools = []
    for t in reg_data["tools"]:
        impl = t.get("implementation", {})
        backend = impl.get("backend", "unknown")
        model = impl.get("model", "")
        models = []
        if model:
            models.append(model)
        elif backend == "dinox":
            models.append("DINO-X API")
        elif backend == "whisperx":
            models.append("WhisperX")
        elif backend == "vlm":
            models.append("gemini-3-pro-preview")
        elif backend == "script":
            models.append("script")
        elif backend == "audio":
            models.append("audio-processor")

        reg_tools.append({
            "name": t["name"],
            "description": t["description"],
            "processing_level": t.get("processing_level", ""),
            "input": t.get("input", {}),
            "output": t.get("output", {}),
            "implementation": {
                "type": impl.get("type", "single_model"),
                "models": models,
                "notes": impl.get("notes", ""),
            },
        })

    usage = {}
    for tname, info in tool_usage.items():
        usage[tname] = {
            "frequency": info["frequency"],
            "ability_ids": sorted(list(info["ability_ids"])),
        }

    registry = {"metadata": reg_data["metadata"], "tools": reg_tools, "usage": usage}

    # ── Baseline stats ──
    baseline_correct = sum(1 for q in questions if q.get("baseline") and q["baseline"]["correct"])
    baseline_total = sum(1 for q in questions if q.get("baseline"))

    # ── Build TRAJ_DATA ──
    traj_data = {
        "metadata": {
            "models": MODELS,
            "total_questions": len(questions),
            "successful_trajectories": len(questions) * len(MODELS),
            "baseline": {
                "model": "gemini-3.1-pro (direct)",
                "correct": baseline_correct,
                "total": baseline_total,
                "accuracy": round(baseline_correct / max(baseline_total, 1) * 100, 1),
            },
        },
        "questions": questions,
    }

    # ── Level counts ──
    level_counts = Counter()
    for t in reg_data["tools"]:
        pl = t.get("processing_level", "")
        if pl:
            level_counts[pl] += 1

    # ── Benchmark stats (context length, tool I/O, accuracy, video duration) ──
    bench_stats = compute_bench_stats(
        cases_data, baseline_records, gpt_records, claude_records, gemini_records,
        case_lookup
    )

    # ── Print summary ──
    print(f"Questions: {len(questions)}")
    print(f"Models: {MODELS}")
    for m in MODELS:
        s = model_stats[m]
        print(f"  {m}: {s['correct']}/{s['trajectories']} ({s['accuracy']}%), avg_steps={s['avg_steps']}")
    print(f"  baseline: {baseline_correct}/{baseline_total} ({round(baseline_correct/max(baseline_total,1)*100,1)}%)")
    print(f"Tools: {len(registry['tools'])}")
    print(f"Tools used: {len(per_tool_total)}/{len(registry['tools'])}")

    # ── Generate HTML ──
    html = generate_html(registry, abilities, traj_stats, traj_data, level_counts, bench_stats)

    out_path = os.path.join(VIEWER_DIR, "pages/09-tool-pool-v12.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nWritten to {out_path}")
    print(f"Size: {len(html):,} bytes")


def generate_html(registry, abilities, traj_stats, traj_data, level_counts, bench_stats):
    reg_json = json.dumps(registry, ensure_ascii=False, separators=(",", ":"))
    abilities_json = json.dumps(abilities, ensure_ascii=False, separators=(",", ":"))
    stats_json = json.dumps(traj_stats, ensure_ascii=False, separators=(",", ":"))
    data_json = json.dumps(traj_data, ensure_ascii=False, separators=(",", ":"))
    bench_stats_json = json.dumps(bench_stats, ensure_ascii=False, separators=(",", ":"))

    total_tools = len(registry["tools"])
    img_count = level_counts.get("image", 0)
    seq_count = level_counts.get("sequence", 0)
    audio_count = level_counts.get("audio", 0)
    total_q = traj_data["metadata"]["total_questions"]

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tool Pool v12 &amp; 3-Model Trajectories — Video Understanding</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --color-text: #1a1a1a;
            --color-text-secondary: #555;
            --color-text-muted: #888;
            --color-bg: #ffffff;
            --color-bg-alt: #f8f9fa;
            --color-border: #e5e5e5;
            --color-accent: #2563eb;
            --color-accent-light: #dbeafe;
            --color-correct: #16a34a;
            --color-correct-bg: #dcfce7;
            --color-wrong: #dc2626;
            --color-wrong-bg: #fef2f2;
            --color-gpt: #10a37f;
            --color-claude: #c96442;
            --color-gemini: #4285f4;
            --font-serif: 'Crimson Pro', 'Noto Sans SC', serif;
            --font-sans: 'Noto Sans SC', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --cat-hlr: #F44336; --cat-tr: #2196F3; --cat-sr: #FF9800;
            --cat-av: #9C27B0; --cat-vp: #4CAF50;
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html {{ scroll-behavior:smooth; }}
        body {{ font-family:var(--font-sans); font-size:16px; line-height:1.6; color:var(--color-text); background:var(--color-bg); }}

        .header {{ background:linear-gradient(135deg,#f8fafc 0%,#fff 100%); border-bottom:1px solid var(--color-border); padding:2.5rem 2rem 2rem; text-align:center; }}
        .header-tag {{ display:inline-block; font-size:0.75rem; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; color:var(--color-accent); background:var(--color-accent-light); padding:0.25rem 0.75rem; border-radius:2rem; margin-bottom:1rem; }}
        .header h1 {{ font-family:var(--font-serif); font-size:clamp(1.5rem,3.5vw,2.25rem); font-weight:700; margin-bottom:0.5rem; }}
        .header .subtitle {{ font-size:0.95rem; color:var(--color-text-secondary); }}

        .main {{ max-width:1200px; margin:0 auto; padding:2rem 1.5rem 4rem; }}
        .section {{ margin-bottom:3rem; }}
        .section-header {{ display:flex; align-items:center; gap:0.75rem; margin-bottom:1.5rem; padding-bottom:0.5rem; border-bottom:2px solid var(--color-text); }}
        .section-number {{ font-family:var(--font-serif); font-size:1.75rem; font-weight:700; color:var(--color-accent); line-height:1; }}
        .section-title {{ font-family:var(--font-serif); font-size:1.25rem; font-weight:600; }}
        .section-count {{ margin-left:auto; font-size:0.875rem; color:var(--color-text-muted); }}

        .stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1rem; margin-bottom:1.5rem; }}
        .stat-card {{ background:var(--color-bg-alt); border:1px solid var(--color-border); border-radius:8px; padding:1rem 1.25rem; text-align:center; }}
        .stat-value {{ font-family:var(--font-serif); font-size:1.75rem; font-weight:700; color:var(--color-accent); }}
        .stat-label {{ font-size:0.8rem; color:var(--color-text-muted); margin-top:0.25rem; }}
        .stat-desc {{ font-size:0.65rem; color:var(--color-text-muted); margin-top:0.15rem; opacity:0.7; }}

        .ability-overview-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:1px; background:var(--color-border); border:1px solid var(--color-border); border-radius:8px; overflow:hidden; }}
        @media (max-width:1000px) {{ .ability-overview-grid {{ grid-template-columns:repeat(3,1fr); }} }}
        @media (max-width:600px) {{ .ability-overview-grid {{ grid-template-columns:1fr 1fr; }} }}
        .ability-col {{ background:var(--color-bg); display:flex; flex-direction:column; }}
        .ability-col-header {{ padding:0.7rem 0.6rem; text-align:center; border-bottom:1px solid var(--color-border); }}
        .ability-col-name {{ font-weight:600; font-size:0.85rem; }}
        .ability-col-sub {{ font-size:0.65rem; color:var(--color-text-muted); margin-top:0.15rem; }}
        .ability-item {{ border-bottom:1px solid #f0f0f0; }}
        .ability-item:last-child {{ border-bottom:none; }}
        .ability-item-header {{ padding:0.4rem 0.55rem; display:flex; align-items:center; gap:0.3rem; font-size:0.8rem; cursor:pointer; }}
        .ability-item-header:hover {{ background:#fafbfd; }}
        .ability-item-id {{ font-family:var(--font-mono); font-size:0.62rem; color:var(--color-accent); font-weight:700; min-width:1.2rem; text-align:right; flex-shrink:0; }}
        .ability-item-name {{ font-size:0.74rem; font-weight:500; line-height:1.3; }}
        .ability-item-count {{ margin-left:auto; font-size:0.6rem; color:var(--color-text-muted); flex-shrink:0; }}

        .tool-filter-bar {{ display:flex; gap:0.5rem; margin-bottom:1.5rem; flex-wrap:wrap; }}
        .filter-btn {{ padding:0.35rem 0.8rem; border:1px solid var(--color-border); border-radius:2rem; background:white; cursor:pointer; font-size:0.78rem; font-family:var(--font-sans); color:var(--color-text-secondary); transition:all 0.15s; }}
        .filter-btn:hover {{ border-color:var(--color-accent); color:var(--color-accent); }}
        .filter-btn.active {{ background:var(--color-accent); color:white; border-color:var(--color-accent); }}

        .tool-card {{ border:1px solid var(--color-border); border-radius:8px; margin-bottom:0.75rem; overflow:hidden; transition:box-shadow 0.15s; }}
        .tool-card:hover {{ box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
        .tool-card-header {{ display:flex; align-items:center; gap:0.6rem; padding:0.65rem 1rem; cursor:pointer; user-select:none; }}
        .tool-card-header:hover {{ background:#fafbfd; }}
        .tool-card-rank {{ font-family:var(--font-mono); font-size:0.62rem; color:var(--color-text-muted); min-width:1.5rem; text-align:right; }}
        .tool-card-name {{ font-family:var(--font-mono); font-size:0.82rem; font-weight:600; color:var(--color-accent); }}
        .type-badge {{ font-size:0.6rem; font-weight:600; padding:0.12rem 0.4rem; border-radius:3px; white-space:nowrap; }}
        .type-badge.single_model {{ background:#dbeafe; color:#1d4ed8; }}
        .type-badge.pipeline {{ background:#fef3c7; color:#92400e; }}
        .type-badge.script {{ background:#d1fae5; color:#065f46; }}
        .level-badge {{ font-size:0.58rem; font-weight:600; padding:0.1rem 0.35rem; border-radius:3px; white-space:nowrap; text-transform:uppercase; letter-spacing:0.03em; }}
        .level-badge.image {{ background:#fce7f3; color:#9d174d; }}
        .level-badge.sequence {{ background:#e0e7ff; color:#3730a3; }}
        .level-badge.audio {{ background:#fef9c3; color:#854d0e; }}
        .tool-card-desc {{ flex:1; font-size:0.72rem; color:var(--color-text-secondary); line-height:1.4; }}
        .tool-card-freq {{ display:flex; align-items:center; gap:0.3rem; margin-left:auto; flex-shrink:0; }}
        .freq-bar {{ width:60px; height:5px; background:#eee; border-radius:3px; overflow:hidden; }}
        .freq-fill {{ height:100%; border-radius:3px; }}
        .freq-num {{ font-family:var(--font-mono); font-size:0.65rem; color:var(--color-text-muted); min-width:2rem; text-align:right; }}
        .tool-card-arrow {{ font-size:0.6rem; color:var(--color-text-muted); transition:transform 0.2s; }}
        .tool-card.expanded .tool-card-arrow {{ transform:rotate(90deg); }}
        .tool-card-body {{ display:none; border-top:1px solid var(--color-border); }}
        .tool-card.expanded .tool-card-body {{ display:block; }}
        .tool-detail {{ padding:1rem; }}
        .tool-detail-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
        @media (max-width:700px) {{ .tool-detail-row {{ grid-template-columns:1fr; }} }}
        .detail-label {{ font-size:0.68rem; font-weight:600; color:var(--color-text-muted); margin-bottom:0.35rem; text-transform:uppercase; letter-spacing:0.04em; }}
        .detail-content {{ font-size:0.75rem; color:var(--color-text-secondary); line-height:1.6; }}
        .io-table {{ width:100%; border-collapse:collapse; font-size:0.72rem; }}
        .io-table th {{ text-align:left; font-size:0.65rem; font-weight:600; color:var(--color-text-muted); padding:0.2rem 0.4rem; border-bottom:1px solid var(--color-border); }}
        .io-table td {{ padding:0.25rem 0.4rem; border-bottom:1px solid #f0f0f0; vertical-align:top; }}
        .io-key {{ font-family:var(--font-mono); font-weight:600; font-size:0.7rem; white-space:nowrap; }}
        .io-type {{ font-family:var(--font-mono); font-size:0.62rem; color:var(--color-text-muted); word-break:break-all; }}
        .model-tag {{ display:inline-block; padding:0.1rem 0.35rem; border-radius:3px; font-size:0.63rem; font-weight:500; margin:0.1rem; }}
        .impl-notes {{ font-size:0.72rem; color:var(--color-text-secondary); line-height:1.6; margin-top:0.5rem; padding:0.6rem 0.8rem; background:#f8f9fb; border-radius:6px; border:1px solid var(--color-border); }}
        .ability-tags {{ display:flex; flex-wrap:wrap; gap:0.2rem; margin-top:0.3rem; }}
        .ability-tag {{ font-size:0.6rem; padding:0.1rem 0.3rem; border-radius:3px; font-weight:500; cursor:pointer; }}
        .ability-tag:hover {{ opacity:0.8; }}
        .traj-freq-bar {{ display:flex; gap:2px; align-items:center; margin-top:0.3rem; }}
        .traj-freq-seg {{ height:14px; border-radius:2px; font-size:0.55rem; font-weight:600; color:white; display:flex; align-items:center; justify-content:center; min-width:16px; }}

        .model-compare-table {{ width:100%; border-collapse:collapse; font-size:0.78rem; }}
        .model-compare-table th {{ padding:0.5rem 0.75rem; background:var(--color-bg-alt); font-weight:600; font-size:0.72rem; text-align:left; border-bottom:1px solid var(--color-border); }}
        .model-compare-table td {{ padding:0.45rem 0.75rem; border-bottom:1px solid #f0f0f0; }}
        .model-compare-table tr:hover {{ background:#fafbfd; }}
        .model-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:0.4rem; vertical-align:middle; }}
        .model-dot.gpt {{ background:var(--color-gpt); }}
        .model-dot.claude {{ background:var(--color-claude); }}
        .model-dot.gemini {{ background:var(--color-gemini); }}

        /* ═══ Section 05: 3-column model comparison ═══ */
        .ability-group {{ margin-bottom:2.5rem; }}
        .ability-group-header {{ display:flex; align-items:center; gap:0.6rem; padding:0.6rem 0.8rem; border-radius:6px; margin-bottom:1rem; cursor:pointer; user-select:none; }}
        .ability-group-header:hover {{ opacity:0.9; }}
        .ability-group-id {{ font-family:var(--font-mono); font-size:0.75rem; font-weight:700; color:white; background:rgba(255,255,255,0.25); padding:0.1rem 0.4rem; border-radius:3px; }}
        .ability-group-name {{ font-weight:600; font-size:0.9rem; color:white; }}
        .ability-group-count {{ margin-left:auto; font-size:0.72rem; color:rgba(255,255,255,0.8); }}
        .ability-group-category {{ font-size:0.65rem; color:rgba(255,255,255,0.7); }}
        .ability-group.collapsed .ability-group-questions {{ display:none; }}

        .question-card {{ border:1px solid var(--color-border); border-radius:8px; margin-bottom:1rem; overflow:hidden; }}
        .question-card-header {{ padding:0.8rem 1rem; border-bottom:1px solid var(--color-border); background:var(--color-bg-alt); }}
        .question-info {{ display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem; flex-wrap:wrap; }}
        .question-benchmark {{ font-size:0.62rem; font-weight:600; padding:0.1rem 0.35rem; border-radius:3px; background:#e0e7ff; color:#3730a3; }}
        .question-id {{ font-family:var(--font-mono); font-size:0.65rem; color:var(--color-text-muted); }}
        .question-seq {{ font-family:var(--font-mono); font-size:0.72rem; font-weight:700; color:var(--color-accent); }}
        .question-text {{ font-size:0.82rem; line-height:1.5; color:var(--color-text); }}

        .model-compare {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0; }}
        .model-column {{ border-right:1px solid var(--color-border); min-height:60px; min-width:0; overflow:hidden; }}
        .model-column:last-child {{ border-right:none; }}
        .model-col-header {{ padding:0.45rem 0.6rem; display:flex; align-items:center; gap:0.4rem; font-size:0.75rem; font-weight:600; border-bottom:1px solid var(--color-border); }}
        .model-col-header.gpt {{ background:#f0fdf4; color:var(--color-gpt); border-top:2px solid var(--color-gpt); }}
        .model-col-header.claude {{ background:#fff7ed; color:var(--color-claude); border-top:2px solid var(--color-claude); }}
        .model-col-header.gemini {{ background:#eff6ff; color:var(--color-gemini); border-top:2px solid var(--color-gemini); }}
        .model-col-header .step-count {{ margin-left:auto; font-size:0.62rem; font-weight:400; color:var(--color-text-muted); }}
        .model-col-header .correct-badge {{ font-size:0.55rem; padding:0.05rem 0.3rem; }}

        .step-list {{ padding:0.4rem; }}
        .tool-step {{ padding:0.35rem 0.5rem; margin-bottom:0.3rem; border-radius:4px; background:var(--color-bg-alt); border:1px solid #f0f0f0; }}
        .tool-step:last-child {{ margin-bottom:0; }}
        .tool-step-header {{ display:flex; align-items:center; gap:0.35rem; }}
        .step-num {{ font-family:var(--font-mono); font-size:0.58rem; font-weight:700; color:white; background:var(--color-accent); width:18px; height:18px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }}
        .step-num.reasoning {{ background:#9ca3af; }}
        .step-name {{ font-family:var(--font-mono); font-size:0.7rem; font-weight:600; color:var(--color-accent); cursor:pointer; }}
        .step-name:hover {{ text-decoration:underline; }}
        .step-name.reasoning {{ color:#6b7280; font-style:italic; cursor:default; }}
        .step-name.reasoning:hover {{ text-decoration:none; }}
        .step-purpose {{ font-size:0.66rem; color:var(--color-text-secondary); line-height:1.4; margin-top:0.2rem; padding-left:1.6rem; }}
        .step-purpose p {{ margin:0.2rem 0; }}
        .step-purpose strong {{ color:var(--color-text); }}
        .step-purpose code {{ font-family:var(--font-mono); font-size:0.6rem; background:var(--color-bg-alt); padding:0.1rem 0.25rem; border-radius:3px; }}
        .step-purpose ul, .step-purpose ol {{ margin:0.2rem 0; padding-left:1.2rem; }}
        .step-purpose li {{ margin:0.1rem 0; }}
        .step-collapsible {{ margin-top:0.25rem; border-radius:4px; border:1px solid #e5e7eb; }}
        .step-collapsible-header {{ display:flex; align-items:center; gap:0.3rem; padding:0.25rem 0.6rem; background:#f3f4f6; cursor:pointer; font-size:0.58rem; font-weight:600; color:#6b7280; text-transform:uppercase; letter-spacing:0.04em; }}
        .step-collapsible-header:hover {{ background:#e5e7eb; }}
        .step-collapsible-toggle {{ font-size:0.5rem; transition:transform 0.15s; }}
        .step-collapsible.expanded .step-collapsible-toggle {{ transform:rotate(90deg); }}
        .step-collapsible-body {{ display:none; padding:0.4rem 0.5rem; font-family:var(--font-mono); font-size:0.6rem; line-height:1.5; color:#4b5563; white-space:pre-wrap; word-break:break-all; max-height:15em; overflow-y:auto; overflow-x:auto; background:#f9fafb; border-top:1px solid #e5e7eb; }}
        .step-collapsible.expanded .step-collapsible-body {{ display:block; }}
        .step-collapsible.thinking .step-collapsible-header {{ background:#f0f0f0; color:#9ca3af; }}
        .step-collapsible.params .step-collapsible-header {{ background:#f0f9ff; color:#3b82f6; }}
        .no-tools {{ padding:1rem; text-align:center; font-size:0.75rem; color:var(--color-text-muted); }}

        /* Tool I/O detail blocks */
        .exec-tool-item {{ display:flex; align-items:center; gap:0.35rem; padding:0.15rem 0 0.15rem 1.6rem; }}
        .exec-tool-dot {{ width:6px; height:6px; border-radius:50%; background:var(--color-accent); flex-shrink:0; }}
        .exec-tool-detail {{ display:none; margin:0.15rem 0 0.3rem 1.6rem; }}
        .exec-tool-detail.expanded {{ display:block; }}
        /* ── Step images (key case thumbnails) ── */
        .step-images {{ display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.4rem 0 0.2rem 1.6rem; }}
        .step-img-thumb {{ cursor:pointer; border:1px solid var(--color-border); border-radius:4px; overflow:hidden; display:flex; flex-direction:column; max-width:140px; transition:box-shadow 0.15s; }}
        .step-img-thumb:hover {{ box-shadow:0 2px 8px rgba(0,0,0,0.15); }}
        .step-img-thumb img {{ width:100%; height:auto; display:block; }}
        .step-img-label {{ font-size:0.55rem; text-align:center; padding:0.15rem 0.25rem; background:var(--color-bg-alt); color:var(--color-text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        /* Lightbox */
        .lightbox-overlay {{ position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.85); z-index:9999; display:flex; align-items:center; justify-content:center; cursor:pointer; }}
        .lightbox-overlay img {{ max-width:90vw; max-height:90vh; border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5); }}

        /* ── Annotations panel ── */
        .ann-panel {{ border:2px solid #f59e0b; border-radius:8px; margin-top:1rem; background:#fffbeb; }}
        .ann-panel-header {{ display:flex; align-items:center; gap:0.5rem; padding:0.6rem 0.8rem; background:#fef3c7; border-bottom:1px solid #f59e0b40; cursor:pointer; border-radius:7px 7px 0 0; }}
        .ann-panel.collapsed .ann-panel-header {{ border-radius:7px; border-bottom:none; }}
        .ann-panel.collapsed .ann-panel-body {{ display:none; }}
        .ann-panel-tag {{ font-size:0.6rem; font-weight:700; padding:0.15rem 0.5rem; border-radius:3px; color:white; }}
        .ann-panel-tag.cat1 {{ background:#ef4444; }}
        .ann-panel-tag.cat2 {{ background:#f59e0b; }}
        .ann-panel-tag.cat3 {{ background:#22c55e; }}
        .ann-panel-title {{ font-size:0.75rem; font-weight:700; color:#92400e; }}
        .ann-panel-toggle {{ font-size:0.6rem; color:#92400e; margin-left:auto; }}
        .ann-panel-body {{ padding:0.6rem 0.8rem; font-size:0.72rem; line-height:1.6; color:#78350f; }}
        .ann-section {{ margin-bottom:0.6rem; }}
        .ann-section-label {{ font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#b45309; margin-bottom:0.2rem; }}
        .ann-summary {{ font-size:0.72rem; line-height:1.5; }}
        .ann-failure {{ background:#fef2f2; border-left:3px solid #ef4444; padding:0.4rem 0.6rem; border-radius:0 4px 4px 0; margin:0.4rem 0; font-size:0.7rem; color:#991b1b; }}
        .ann-bl-note {{ background:#fff7ed; border-left:3px solid #f59e0b; padding:0.4rem 0.6rem; border-radius:0 4px 4px 0; margin:0.4rem 0; font-size:0.7rem; color:#9a3412; }}
        .ann-model-note {{ background:#eff6ff; border-left:3px solid #3b82f6; padding:0.4rem 0.6rem; border-radius:0 4px 4px 0; margin:0.3rem 0; font-size:0.68rem; color:#1e40af; }}
        .ann-model-note .ann-model-name {{ font-weight:700; }}
        /* Step annotation badge */
        .step-ann {{ display:inline-flex; align-items:flex-start; gap:0.3rem; margin:0.25rem 0 0.1rem 1.6rem; padding:0.3rem 0.5rem; background:#fef3c7; border:1px solid #f59e0b40; border-radius:4px; font-size:0.62rem; line-height:1.4; color:#92400e; }}
        .step-ann::before {{ content:'📝'; font-size:0.55rem; flex-shrink:0; margin-top:0.1rem; }}
        /* Quick filter buttons */
        .quick-filter-bar {{ display:flex; gap:0.4rem; margin-bottom:0.6rem; align-items:center; }}
        .quick-filter-btn {{ padding:0.3rem 0.7rem; border:2px solid; border-radius:2rem; cursor:pointer; font-size:0.7rem; font-weight:600; font-family:var(--font-sans); transition:all 0.15s; background:white; }}
        .quick-filter-btn:hover {{ opacity:0.8; }}
        .quick-filter-btn.cat1 {{ border-color:#ef4444; color:#ef4444; }}
        .quick-filter-btn.cat1.active {{ background:#ef4444; color:white; }}
        .quick-filter-btn.cat2 {{ border-color:#f59e0b; color:#b45309; }}
        .quick-filter-btn.cat2.active {{ background:#f59e0b; color:white; }}
        .quick-filter-btn.cat3 {{ border-color:#22c55e; color:#22c55e; }}
        .quick-filter-btn.cat3.active {{ background:#22c55e; color:white; }}

        .step-block {{ margin-top:0.3rem; border-radius:4px; padding:0.5rem 0.75rem; font-size:0.68rem; line-height:1.6; position:relative; }}
        .step-block-label {{ display:inline-block; font-size:0.55rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; padding:0.05rem 0.3rem; border-radius:2px; margin-right:0.3rem; vertical-align:middle; }}
        .step-block-content {{ font-family:var(--font-mono); font-size:0.62rem; color:inherit; word-break:break-all; white-space:pre-wrap; max-height:20em; overflow-y:auto; margin-top:0.15rem; }}
        .step-block.input {{ background:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; }}
        .step-block.input .step-block-label {{ background:#bfdbfe; color:#1e3a8a; }}
        .step-block.output {{ background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; }}
        .step-block.output .step-block-label {{ background:#bbf7d0; color:#14532d; }}
        .param-table {{ width:100%; border-collapse:collapse; font-size:0.7rem; }}
        .param-table td {{ padding:0.2rem 0.4rem; border-bottom:1px solid var(--color-border); vertical-align:top; word-break:break-all; }}
        .param-table th {{ font-size:0.65rem; padding:0.25rem 0.4rem; border-bottom:2px solid var(--color-border); text-align:left; }}
        .param-table .param-key {{ font-family:var(--font-mono); font-weight:600; color:var(--color-accent); white-space:nowrap; width:1%; }}
        .param-table .param-val {{ font-family:var(--font-mono); color:var(--color-text); white-space:pre-wrap; }}
        .param-val .param-table {{ margin:0.15rem 0; font-size:0.65rem; background:var(--color-bg-alt); border-radius:4px; }}
        .param-val .param-table td, .param-val .param-table th {{ padding:0.15rem 0.3rem; }}
        .param-prefix-hint {{ font-size:0.55rem; color:var(--color-text-muted); margin-bottom:0.2rem; }}
        .param-prefix-dim {{ color:var(--color-text-muted); }}
        .vid-dur {{ font-size:0.6rem; color:var(--color-text-muted); font-family:var(--font-mono); }}

        /* Collapsible sections */
        .collapsible-header {{ cursor:pointer; user-select:none; }}
        .collapsible-header:hover {{ opacity:0.8; }}
        .toggle-icon {{ margin-left:auto; font-size:0.7rem; color:var(--color-text-muted); transition:transform 0.2s; }}
        .collapsible-header:not(.collapsed) .toggle-icon {{ transform:rotate(180deg); }}
        .collapsible-body.collapsed {{ display:none; }}

        .correct-badge {{ display:inline-flex; align-items:center; gap:0.2rem; font-size:0.62rem; font-weight:600; padding:0.1rem 0.4rem; border-radius:3px; }}
        .correct-badge.correct {{ background:var(--color-correct-bg); color:var(--color-correct); }}
        .correct-badge.wrong {{ background:var(--color-wrong-bg); color:var(--color-wrong); }}
        .answer-detail {{ font-size:0.62rem; color:var(--color-text-muted); font-family:var(--font-mono); }}

        .video-btn {{ display:inline-flex; align-items:center; gap:0.2rem; font-size:0.62rem; font-weight:600; padding:0.15rem 0.45rem; border-radius:3px; border:1px solid; text-decoration:none; cursor:pointer; }}
        a.youtube-btn {{ text-decoration:none; background:#ff000012; border-color:#cc0000; color:#cc0000; }}
        a.youtube-btn:hover {{ background:#ff000020; border-color:#ff0000; color:#ff0000; }}

        /* Baseline row */
        .baseline-row {{ border-top:1px solid var(--color-border); border-bottom:1px solid var(--color-border); }}
        .baseline-row-header {{ padding:0.5rem 0.8rem; font-size:0.75rem; font-weight:600; display:flex; align-items:center; gap:0.4rem; background:#fff7ed; color:#9a3412; border-top:2px solid #FF9800; border-bottom:1px solid var(--color-border); cursor:pointer; user-select:none; }}
        .baseline-row-header:hover {{ background:#fff3e0; }}
        .baseline-body {{ display:none; }}
        .baseline-row.expanded .baseline-body {{ display:block; }}
        .baseline-response {{ font-size:0.72rem; color:var(--color-text-secondary); line-height:1.5; padding:0.6rem 0.8rem; }}
        .baseline-reasoning {{ padding:0.6rem 0.8rem; border-bottom:1px solid var(--color-border); }}
        .baseline-section-label {{ font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:var(--color-text-muted); margin-bottom:0.3rem; }}
        .baseline-md {{ font-size:0.72rem; line-height:1.6; color:var(--color-text-secondary); }}
        .baseline-md p {{ margin:0.3rem 0; }}
        .baseline-md strong {{ color:var(--color-text); }}
        .baseline-md ul, .baseline-md ol {{ margin:0.3rem 0; padding-left:1.2rem; }}
        .baseline-md li {{ margin:0.15rem 0; }}
        .baseline-md code {{ font-family:var(--font-mono); font-size:0.65rem; background:var(--color-bg-alt); padding:0.1rem 0.3rem; border-radius:3px; }}
        .baseline-toggle {{ font-size:0.6rem; color:var(--color-text-muted); margin-left:auto; }}

        .case-filter-bar {{ display:flex; gap:0.4rem; margin-bottom:1.5rem; flex-wrap:wrap; align-items:center; }}
        .case-filter-label {{ font-size:0.72rem; color:var(--color-text-muted); margin-right:0.3rem; }}
        .case-filter-btn {{ padding:0.25rem 0.6rem; border:1px solid var(--color-border); border-radius:2rem; background:white; cursor:pointer; font-size:0.7rem; font-family:var(--font-sans); color:var(--color-text-secondary); transition:all 0.15s; }}
        .case-filter-btn:hover {{ border-color:var(--color-accent); color:var(--color-accent); }}
        .case-filter-btn.active {{ background:var(--color-accent); color:white; border-color:var(--color-accent); }}
        /* Combo tabs: model vs baseline */
        .combo-filter-section {{ margin-bottom:1rem; }}
        .combo-filter-group {{ display:flex; gap:0.35rem; margin-bottom:0.5rem; flex-wrap:wrap; align-items:center; }}
        .combo-filter-group .combo-label {{ font-size:0.72rem; font-weight:600; min-width:140px; }}
        .combo-tab {{ padding:0.3rem 0.7rem; border-radius:2rem; cursor:pointer; font-size:0.7rem; font-weight:600; border:2px solid transparent; background:var(--color-bg-alt); color:var(--color-text-secondary); transition:all 0.15s; display:inline-flex; align-items:center; gap:0.3rem; }}
        .combo-tab:hover {{ opacity:0.85; }}
        .combo-tab .tab-count {{ font-family:var(--font-mono); font-size:0.65rem; }}
        .combo-tab.active {{ color:white; }}
        .combo-tab.both_correct.active {{ background:var(--color-correct); border-color:var(--color-correct); }}
        .combo-tab.bl_only.active {{ background:#FF9800; border-color:#FF9800; }}
        .combo-tab.model_only.active {{ background:#2196F3; border-color:#2196F3; }}
        .combo-tab.both_wrong.active {{ background:var(--color-wrong); border-color:var(--color-wrong); }}
        .combo-tab.combo-all.active {{ background:var(--color-accent); border-color:var(--color-accent); }}
        .combo-divider {{ display:flex; align-items:center; gap:0.5rem; margin:0.6rem 0; font-size:0.68rem; color:var(--color-text-muted); }}
        .combo-divider::before, .combo-divider::after {{ content:''; flex:1; border-top:1px dashed var(--color-border); }}
        /* Independent dropdown filters */
        .corr-filter-row {{ display:flex; gap:0.6rem; margin-bottom:1rem; flex-wrap:wrap; align-items:center; }}
        .corr-filter-row .case-filter-label {{ font-size:0.72rem; color:var(--color-text-muted); margin-right:0.2rem; }}
        .corr-filter-item {{ display:flex; align-items:center; gap:0.25rem; }}
        .corr-filter-item .corr-model-label {{ font-size:0.7rem; font-weight:600; }}
        .corr-filter-item select {{ padding:0.2rem 0.4rem; border:1.5px solid var(--color-border); border-radius:4px; font-size:0.68rem; font-family:var(--font-sans); color:var(--color-text-secondary); background:white; cursor:pointer; outline:none; transition:border-color 0.15s; }}
        .corr-filter-item select:focus {{ border-color:var(--color-accent); }}
        .corr-filter-item select.filter-correct {{ border-color:var(--color-correct); color:var(--color-correct); background:var(--color-correct-bg); }}
        .corr-filter-item select.filter-wrong {{ border-color:var(--color-wrong); color:var(--color-wrong); background:var(--color-wrong-bg); }}

        /* Search */
        .case-search-row {{ margin-bottom:0.8rem; }}
        .case-search-row input {{ width:100%; max-width:400px; padding:0.4rem 0.7rem; border:1.5px solid var(--color-border); border-radius:6px; font-size:0.75rem; font-family:var(--font-sans); color:var(--color-text); outline:none; transition:border-color 0.15s; }}
        .case-search-row input:focus {{ border-color:var(--color-accent); }}
        .case-search-row input::placeholder {{ color:var(--color-text-muted); }}
        /* Pagination */
        .pagination-bar {{ display:flex; align-items:center; justify-content:center; gap:0.3rem; margin:0.8rem 0; flex-wrap:wrap; }}
        .pagination-bar:empty {{ display:none; }}
        .page-btn {{ padding:0.25rem 0.55rem; border:1px solid var(--color-border); border-radius:4px; background:white; cursor:pointer; font-size:0.7rem; font-family:var(--font-sans); color:var(--color-text-secondary); transition:all 0.12s; }}
        .page-btn:hover {{ border-color:var(--color-accent); color:var(--color-accent); }}
        .page-btn.active {{ background:var(--color-accent); color:white; border-color:var(--color-accent); }}
        .page-btn.disabled {{ opacity:0.4; pointer-events:none; }}
        .page-info {{ font-size:0.7rem; color:var(--color-text-muted); margin:0 0.5rem; }}
        .page-size-select {{ padding:0.2rem 0.3rem; border:1px solid var(--color-border); border-radius:4px; font-size:0.68rem; font-family:var(--font-sans); color:var(--color-text-secondary); margin-left:0.5rem; }}

        /* Mobile tabs */
        .model-tabs-wrapper {{ display:none; }}
        .model-tabs {{ display:flex; border-bottom:1px solid var(--color-border); }}
        .model-tab {{ flex:1; padding:0.5rem; text-align:center; font-size:0.75rem; font-weight:600; cursor:pointer; border-bottom:2px solid transparent; color:var(--color-text-muted); }}
        .model-tab.active {{ color:var(--color-accent); border-bottom-color:var(--color-accent); }}
        .model-tab-panel {{ display:none; }}
        .model-tab-panel.active {{ display:block; }}
        @media (max-width:900px) {{
            .model-compare {{ display:none; }}
            .model-tabs-wrapper {{ display:block !important; }}
        }}

        #back-top {{ position:fixed; bottom:2rem; right:2rem; display:none; padding:0.5rem 1rem; background:var(--color-accent); color:white; border:none; border-radius:2rem; font-family:var(--font-sans); font-size:0.78rem; font-weight:500; cursor:pointer; box-shadow:0 2px 12px rgba(37,99,235,0.3); z-index:1000; }}
        #back-top:hover {{ box-shadow:0 4px 16px rgba(37,99,235,0.4); transform:translateY(-1px); }}
        #back-btn {{ position:fixed; bottom:2rem; left:2rem; display:none; padding:0.5rem 1rem; background:#6b7280; color:white; border:none; border-radius:2rem; font-family:var(--font-sans); font-size:0.78rem; font-weight:500; cursor:pointer; box-shadow:0 2px 12px rgba(0,0,0,0.15); z-index:1000; }}
        #back-btn:hover {{ background:#4b5563; }}
        @keyframes flash-highlight {{ 0% {{ box-shadow:0 0 0 3px rgba(37,99,235,0.4); }} 100% {{ box-shadow:0 0 0 0 rgba(37,99,235,0); }} }}
        .flash {{ animation:flash-highlight 1.5s ease-out; }}
    </style>
</head>
<body>
<header class="header">
    <div class="header-tag">Tool Pool v12 + 3-Model Comparison</div>
    <h1>视频理解工具池 v12 &amp; 三模型轨迹对比</h1>
    <p class="subtitle">{total_tools} Tools ({img_count} Image &middot; {seq_count} Sequence &middot; {audio_count} Audio) &times; 22 Abilities &times; 3 Models &times; {total_q} Questions</p>
</header>

<main class="main">
    <section class="section" id="sec-overview">
        <div class="section-header"><span class="section-number">01</span><span class="section-title">概览</span></div>
        <div class="stats-grid" id="stats-grid"></div>
    </section>
    <section class="section" id="sec-abilities">
        <div class="section-header"><span class="section-number">02</span><span class="section-title">能力框架</span><span class="section-count" id="abilities-count"></span></div>
        <div id="abilities-container"></div>
    </section>
    <section class="section" id="sec-tools">
        <div class="section-header collapsible-header collapsed" onclick="toggleSection('tools')"><span class="section-number">03</span><span class="section-title">工具池</span><span class="section-count" id="tools-count"></span><span class="toggle-icon">&#9660;</span></div>
        <div class="collapsible-body collapsed" id="tools-body">
            <div class="tool-filter-bar" id="tool-filter"></div>
            <div id="tools-container"></div>
        </div>
    </section>
    <section class="section" id="sec-trajectories">
        <div class="section-header collapsible-header collapsed" onclick="toggleSection('trajectories')"><span class="section-number">04</span><span class="section-title">轨迹统计</span><span class="section-count" id="traj-stats-count"></span><span class="toggle-icon">&#9660;</span></div>
        <div class="collapsible-body collapsed" id="trajectories-body">
            <div id="traj-container"></div>
        </div>
    </section>
    <section class="section" id="sec-bench">
        <div class="section-header"><span class="section-number">05</span><span class="section-title">Benchmark 统计</span><span class="section-count">3 Benchmarks &times; 4 Models</span></div>
        <div id="bench-container"></div>
    </section>

    <section class="section" id="sec-cases">
        <div class="section-header"><span class="section-number">06</span><span class="section-title">轨迹详情</span><span class="section-count" id="cases-count"></span></div>
        <div class="case-search-row">
            <input type="text" id="case-search" placeholder="搜索题目关键词或序号（如 #12）..." oninput="applyAllFilters()">
        </div>
        <div class="case-filter-bar" id="case-filter"></div>
        <div class="quick-filter-bar" id="quick-filter-bar"></div>
        <div class="combo-filter-section" id="combo-filter-section"></div>
        <div class="combo-divider">或 自定义组合</div>
        <div class="corr-filter-row" id="corr-filter-row"></div>
        <div class="pagination-bar" id="pagination-top"></div>
        <div id="cases-container"></div>
        <div class="pagination-bar" id="pagination-bottom"></div>
    </section>
</main>

<button id="back-top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">&#8593; Top</button>
<button id="back-btn" onclick="goBack()" style="display:none">&#8592; Back</button>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const REGISTRY = {reg_json};
const ABILITIES = {abilities_json};
const TRAJ_STATS = {stats_json};
const TRAJ_DATA = {data_json};
const BENCH_STATS = {bench_stats_json};

const CATEGORY_COLORS = {{'High-Level Reasoning':'#F44336','Temporal Relation':'#2196F3','Spatial Relation':'#FF9800','Audio-Visual':'#9C27B0','Visual Perception':'#4CAF50'}};
const MODEL_COLORS = {{'gpt-5.4':'#10a37f','claude-opus-4-6':'#c96442','gemini':'#4285f4'}};
const MODEL_SHORT = {{'gpt-5.4':'GPT-5.4','claude-opus-4-6':'Claude Opus 4.6','gemini':'Gemini 3.0 Pro'}};
const MODEL_CSS = {{'gpt-5.4':'gpt','claude-opus-4-6':'claude','gemini':'gemini'}};
const TYPE_LABELS = {{'single_model':'Single Model','pipeline':'Pipeline','script':'Script'}};
const LEVEL_LABELS = {{'image':'Image','sequence':'Sequence','audio':'Audio'}};
const LEVEL_COLORS = {{'image':'#9d174d','sequence':'#3730a3','audio':'#854d0e'}};

const abilityMap = {{}};
ABILITIES.forEach(a => {{ abilityMap[a.ability_id] = a; }});
const tools = REGISTRY.tools;
const usage = REGISTRY.usage;
const toolOrder = [...tools].sort((a,b) => (TRAJ_STATS.per_tool_total[b.name]||0) - (TRAJ_STATS.per_tool_total[a.name]||0));
const catOrder = ['High-Level Reasoning','Temporal Relation','Spatial Relation','Audio-Visual','Visual Perception'];
const abilitiesByCategory = {{}};
catOrder.forEach(c => {{ abilitiesByCategory[c] = []; }});
ABILITIES.forEach(a => {{ if(abilitiesByCategory[a.ability_category]) abilitiesByCategory[a.ability_category].push(a); }});
const typeCounts = {{single_model:0,pipeline:0,script:0}};
tools.forEach(t => {{ typeCounts[t.implementation.type]++; }});
const levelCounts = {{}};
tools.forEach(t => {{ if(t.processing_level) levelCounts[t.processing_level]=(levelCounts[t.processing_level]||0)+1; }});
const questionsPerAbility = {{}};
TRAJ_DATA.questions.forEach(q => {{ questionsPerAbility[q.ability_id] = (questionsPerAbility[q.ability_id]||0)+1; }});
const questionsByAbility = {{}};
TRAJ_DATA.questions.forEach(q => {{ if(!questionsByAbility[q.ability_id]) questionsByAbility[q.ability_id]=[]; questionsByAbility[q.ability_id].push(q); }});

const navHistory = [];
function jumpTo(id) {{ navHistory.push(window.scrollY); document.getElementById('back-btn').style.display='block'; const el=document.getElementById(id); if(!el) return; if(el.classList.contains('tool-card')&&!el.classList.contains('expanded')) el.classList.add('expanded'); if(el.classList.contains('ability-group')&&el.classList.contains('collapsed')) el.classList.remove('collapsed'); el.scrollIntoView({{behavior:'smooth',block:'start'}}); el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }}
function goBack() {{ if(!navHistory.length) return; window.scrollTo({{top:navHistory.pop(),behavior:'smooth'}}); if(!navHistory.length) document.getElementById('back-btn').style.display='none'; }}

function init() {{ renderOverview(); renderAbilities(); renderToolPool(); renderTrajectories(); renderBenchStats(); renderCases(); window.addEventListener('scroll',()=>{{ document.getElementById('back-top').style.display=window.scrollY>400?'block':'none'; }}); }}

function renderOverview() {{
    const models = TRAJ_DATA.metadata.models;
    const totalSteps = models.reduce((s,m) => s + (TRAJ_STATS.models[m]||{{}}).total_steps||0, 0);
    const usedTools = new Set(); Object.keys(TRAJ_STATS.per_tool_total).forEach(t => {{ if(TRAJ_STATS.per_tool_total[t]>0) usedTools.add(t); }});
    const totalQ = TRAJ_DATA.metadata.total_questions;
    const cards = [
        {{v:tools.length,l:'Tools'}},
        {{v:ABILITIES.length,l:'Abilities'}},
        {{v:models.length,l:'Models'}},
        {{v:totalQ,l:'Questions'}},
        {{v:`${{totalQ*models.length}}`,l:'Trajectories'}},
        {{v:totalSteps,l:'Total Steps'}},
        {{v:usedTools.size+'/'+tools.length,l:'Tools Used'}},
    ];
    // Baseline accuracy
    const bl = TRAJ_DATA.metadata.baseline;
    if(bl) cards.push({{v:`${{bl.correct}}/${{bl.total}}`,l:`Baseline ${{bl.accuracy}}%`}});
    // Per-model accuracy cards
    models.forEach(m => {{
        const s = TRAJ_STATS.models[m];
        if(s) cards.push({{v:`${{s.correct}}/${{s.trajectories}}`,l:`${{MODEL_SHORT[m]}} ${{s.accuracy}}%`}});
    }});
    document.getElementById('stats-grid').innerHTML = cards.map(c => `<div class="stat-card"><div class="stat-value">${{c.v}}</div><div class="stat-label">${{c.l}}</div></div>`).join('');
}}

function renderAbilities() {{
    document.getElementById('abilities-count').textContent = `${{ABILITIES.length}} Abilities · ${{catOrder.length}} Categories`;
    let html = '<div class="ability-overview-grid">';
    catOrder.forEach(cat => {{
        const color = CATEGORY_COLORS[cat]; const abs = abilitiesByCategory[cat];
        html += `<div class="ability-col"><div class="ability-col-header" style="border-top:3px solid ${{color}}"><div class="ability-col-name" style="color:${{color}}">${{cat}}</div><div class="ability-col-sub">${{abs.length}} abilities</div></div>`;
        abs.forEach(a => {{
            const qCount = questionsPerAbility[a.ability_id]||0;
            html += `<div class="ability-item" id="ability-${{a.ability_id}}"><div class="ability-item-header" onclick="jumpTo('ag-${{a.ability_id}}')"><span class="ability-item-id">${{a.ability_id}}</span><span class="ability-item-name">${{a.ability_name}}</span><span class="ability-item-count">${{qCount}}q</span></div></div>`;
        }});
        html += '</div>';
    }});
    html += '</div>';
    document.getElementById('abilities-container').innerHTML = html;
}}

function renderToolPool() {{
    document.getElementById('tools-count').textContent = `${{tools.length}} Tools`;
    const models = TRAJ_DATA.metadata.models;
    let fh = `<button class="filter-btn active" onclick="filterTools('all','all',this)">All (${{tools.length}})</button><span style="color:var(--color-border);margin:0 0.2rem">|</span>`;
    Object.entries(typeCounts).forEach(([t,c]) => {{ if(c>0) fh += `<button class="filter-btn" onclick="filterTools('type','${{t}}',this)">${{TYPE_LABELS[t]}} (${{c}})</button>`; }});
    fh += `<span style="color:var(--color-border);margin:0 0.2rem">|</span>`;
    Object.entries(levelCounts).forEach(([l,c]) => {{ fh += `<button class="filter-btn" onclick="filterTools('level','${{l}}',this)" style="border-color:${{LEVEL_COLORS[l]||'#888'}}30">${{LEVEL_LABELS[l]||l}} (${{c}})</button>`; }});
    document.getElementById('tool-filter').innerHTML = fh;

    let html = '';
    toolOrder.forEach((tool, idx) => {{
        const u = usage[tool.name]||{{frequency:0,ability_ids:[]}}; const trajTotal = TRAJ_STATS.per_tool_total[tool.name]||0;
        const maxTF = Math.max(...Object.values(TRAJ_STATS.per_tool_total),1); const pct = Math.round((trajTotal/maxTF)*100);
        const impl = tool.implementation; const level = tool.processing_level||''; const perModel = TRAJ_STATS.per_tool[tool.name]||{{}};

        html += `<div class="tool-card" data-type="${{impl.type}}" data-level="${{level}}" id="tool-${{tool.name}}">`;
        html += `<div class="tool-card-header" onclick="toggleToolCard(this)"><span class="tool-card-rank">${{idx+1}}</span><span class="tool-card-name">${{tool.name}}</span><span class="type-badge ${{impl.type}}">${{TYPE_LABELS[impl.type]}}</span>`;
        if(level) html += `<span class="level-badge ${{level}}">${{LEVEL_LABELS[level]||level}}</span>`;
        html += `<span class="tool-card-desc">${{tool.description}}</span><span class="tool-card-freq"><span class="freq-bar"><span class="freq-fill" style="width:${{pct}}%;background:var(--color-accent)"></span></span><span class="freq-num">${{trajTotal}}&times;</span></span><span class="tool-card-arrow">&#9654;</span></div>`;

        html += `<div class="tool-card-body"><div class="tool-detail">`;
        html += `<div style="margin-bottom:1rem"><div class="detail-label">Description</div><div class="detail-content">${{tool.description}}</div></div>`;
        html += `<div class="tool-detail-row"><div class="detail-block"><div class="detail-label">Input</div><table class="io-table"><tr><th>Parameter</th><th>Type</th></tr>`;
        Object.entries(tool.input).forEach(([k,v]) => {{ html += `<tr><td class="io-key">${{k}}</td><td class="io-type">${{escHtml(String(v))}}</td></tr>`; }});
        html += `</table></div><div class="detail-block"><div class="detail-label">Output</div><table class="io-table"><tr><th>Field</th><th>Type</th></tr>`;
        Object.entries(tool.output).forEach(([k,v]) => {{ html += `<tr><td class="io-key">${{k}}</td><td class="io-type">${{escHtml(String(v))}}</td></tr>`; }});
        html += `</table></div></div>`;

        html += `<div style="margin-bottom:1rem"><div class="detail-label">Implementation · ${{TYPE_LABELS[impl.type]}}${{level?' · <span class="level-badge '+level+'" style="vertical-align:middle">'+(LEVEL_LABELS[level]||level)+'</span>':''}}</div>`;
        html += `<div style="margin-top:0.3rem"><strong style="font-size:0.7rem;color:var(--color-text-muted)">Backend:</strong> `;
        impl.models.forEach(m => {{ html += `<span class="model-tag" style="background:#eef2ff;color:#3730a3">${{m}}</span> `; }});
        html += `</div>`;
        if(impl.notes) html += `<div class="impl-notes">${{escHtml(impl.notes)}}</div>`;
        html += `</div>`;

        html += `<div class="tool-detail-row"><div class="detail-block"><div class="detail-label">Trajectory Usage (per model)</div><div class="traj-freq-bar">`;
        models.forEach(m => {{ const cnt=perModel[m]||0; const w=Math.max(cnt/1.5,16); html += `<span class="traj-freq-seg" style="width:${{w}}px;background:${{MODEL_COLORS[m]||'#888'}}" title="${{MODEL_SHORT[m]||m}}: ${{cnt}}">${{cnt}}</span>`; }});
        html += `</div><div style="font-size:0.6rem;color:var(--color-text-muted);margin-top:0.2rem">`;
        models.forEach(m => {{ html += `<span class="model-dot ${{MODEL_CSS[m]||'gemini'}}"></span>${{MODEL_SHORT[m]||m}} `; }});
        html += `</div></div>`;
        html += `<div class="detail-block"><div class="detail-label">Ability Coverage (${{(u.ability_ids||[]).length}} abilities)</div><div class="ability-tags">`;
        (u.ability_ids||[]).forEach(aid => {{ const ab=abilityMap[aid]; if(!ab) return; const color=CATEGORY_COLORS[ab.ability_category]||'#888'; html += `<span class="ability-tag" style="background:${{color}}15;color:${{color}}" onclick="jumpTo('ability-${{aid}}')">${{aid}} ${{ab.ability_name}}</span>`; }});
        html += `</div></div></div></div></div></div>`;
    }});
    document.getElementById('tools-container').innerHTML = html;
}}

function renderTrajectories() {{
    const models = TRAJ_DATA.metadata.models;
    document.getElementById('traj-stats-count').textContent = `${{TRAJ_DATA.metadata.total_questions * models.length}} Trajectories`;
    let html = '';
    html += `<div style="margin-bottom:2rem"><div class="detail-label" style="margin-bottom:0.5rem">模型对比</div>`;
    html += `<table class="model-compare-table"><thead><tr><th>Model</th><th>Accuracy</th><th>Avg Steps</th><th>Total Steps</th><th>Tools Used</th><th>Trajectories</th></tr></thead><tbody>`;
    // Baseline row
    const bl = TRAJ_DATA.metadata.baseline;
    if(bl) html += `<tr style="background:#fff7ed"><td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#FF9800;margin-right:0.4rem;vertical-align:middle"></span><strong>${{bl.model}}</strong></td><td><strong>${{bl.accuracy}}%</strong> (${{bl.correct}}/${{bl.total}})</td><td>—</td><td>—</td><td>—</td><td>${{bl.total}}</td></tr>`;
    models.forEach(m => {{
        const s = TRAJ_STATS.models[m]; if(!s) return;
        html += `<tr><td><span class="model-dot ${{MODEL_CSS[m]||'gemini'}}"></span><strong>${{MODEL_SHORT[m]||m}}</strong></td>`;
        html += `<td><strong>${{s.accuracy}}%</strong> (${{s.correct}}/${{s.trajectories}})</td><td><strong>${{s.avg_steps}}</strong></td><td>${{s.total_steps}}</td><td>${{s.total_tools_used}}/${{tools.length}}</td><td>${{s.trajectories}}</td></tr>`;
    }});
    html += `</tbody></table></div>`;

    const sortedTools = Object.entries(TRAJ_STATS.per_tool_total).sort((a,b) => b[1]-a[1]);
    html += `<div><div class="detail-label" style="margin-bottom:0.5rem">工具使用频次（按轨迹统计）</div>`;
    html += `<div style="overflow-x:auto"><table class="model-compare-table" style="font-size:0.72rem"><thead><tr><th>#</th><th>Tool</th>`;
    models.forEach(m => {{ html += `<th><span class="model-dot ${{MODEL_CSS[m]||'gemini'}}"></span>${{MODEL_SHORT[m]||m}}</th>`; }});
    html += `<th>Total</th><th style="width:120px">Distribution</th></tr></thead><tbody>`;
    sortedTools.forEach(([name,total],i) => {{
        const pm = TRAJ_STATS.per_tool[name]||{{}};
        html += `<tr><td style="color:var(--color-text-muted)">${{i+1}}</td><td><a style="font-family:var(--font-mono);font-size:0.72rem;color:var(--color-accent);cursor:pointer;text-decoration:none" onclick="jumpTo('tool-${{name}}')">${{name}}</a></td>`;
        models.forEach(m => {{ html += `<td>${{pm[m]||0}}</td>`; }});
        html += `<td><strong>${{total}}</strong></td><td><div style="display:flex;gap:1px;height:12px">`;
        models.forEach(m => {{ const w=Math.round(((pm[m]||0)/Math.max(total,1))*100); html += `<div style="width:${{w}}%;background:${{MODEL_COLORS[m]||'#888'}};border-radius:2px" title="${{MODEL_SHORT[m]||m}}:${{pm[m]||0}}"></div>`; }});
        html += `</div></td></tr>`;
    }});
    tools.forEach(t => {{ if(!TRAJ_STATS.per_tool_total[t.name]) {{ html += `<tr style="opacity:0.4"><td style="color:var(--color-text-muted)">-</td><td><span style="font-family:var(--font-mono);font-size:0.72rem">${{t.name}}</span></td>`; models.forEach(()=>{{html+=`<td>0</td>`}}); html += `<td>0</td><td>-</td></tr>`; }} }});
    html += `</tbody></table></div></div>`;
    document.getElementById('traj-container').innerHTML = html;
}}

// ── Pagination state ──
let casePage = 1;
let casePageSize = 1;
let caseFilteredIds = []; // sequential_ids of filtered questions

function renderCases() {{
    const models = TRAJ_DATA.metadata.models;
    const totalQ = TRAJ_DATA.questions.length;

    // Category + Benchmark filter bar
    let fh = `<span class="case-filter-label">Filter:</span><button class="case-filter-btn active" data-type="cat" data-val="all" onclick="setCaseFilter('cat','all',this)">All (${{totalQ}})</button>`;
    catOrder.forEach(cat => {{
        const color = CATEGORY_COLORS[cat]; const abs = abilitiesByCategory[cat]||[]; let count=0;
        abs.forEach(a => {{ count += (questionsPerAbility[a.ability_id]||0); }});
        fh += `<button class="case-filter-btn" data-type="cat" data-val="${{cat}}" onclick="setCaseFilter('cat','${{cat}}',this)" style="border-color:${{color}}40">${{cat.split(' ')[0]}} (${{count}})</button>`;
    }});
    const benchCounts = {{}}; TRAJ_DATA.questions.forEach(q => {{ benchCounts[q.benchmark]=(benchCounts[q.benchmark]||0)+1; }});
    fh += `<span style="color:var(--color-border);margin:0 0.3rem">|</span>`;
    Object.entries(benchCounts).forEach(([b,c]) => {{ fh += `<button class="case-filter-btn" data-type="bench" data-val="${{b}}" onclick="setCaseFilter('bench','${{b}}',this)">${{b}} (${{c}})</button>`; }});
    document.getElementById('case-filter').innerHTML = fh;

    // Build flat question list with filter metadata
    window._allCaseItems = [];
    ABILITIES.forEach(a => {{
        const qs = questionsByAbility[a.ability_id]; if(!qs||!qs.length) return;
        qs.forEach(q => {{
            const blOk = q.baseline ? (q.baseline.correct?1:0) : -1;
            const gptOk = (q.per_model&&q.per_model['gpt-5.4']) ? (q.per_model['gpt-5.4'].correct?1:0) : -1;
            const clOk = (q.per_model&&q.per_model['claude-opus-4-6']) ? (q.per_model['claude-opus-4-6'].correct?1:0) : -1;
            const gemOk = (q.per_model&&q.per_model['gemini']) ? (q.per_model['gemini'].correct?1:0) : -1;
            window._allCaseItems.push({{
                q, a,
                bl: blOk, gpt: gptOk, claude: clOk, gem: gemOk,
                cat: a.ability_category,
                bench: q.benchmark,
                searchText: `#${{q.sequential_id}} ${{q.question_text}} ${{q.question_id}}`.toLowerCase(),
            }});
        }});
    }});

    // Combo tabs: each model vs Baseline
    const comboModels = [
        {{key:'gpt', label:'GPT-5.4', color:'#10a37f'}},
        {{key:'claude', label:'Claude Opus 4.6', color:'#c96442'}},
        {{key:'gem', label:'Gemini 3.0 Pro', color:'#4285f4'}},
    ];
    const comboCats = ['both_correct','bl_only','model_only','both_wrong'];
    const comboCatLabels = {{both_correct:'Both Correct', bl_only:'Baseline Only', model_only:'{{MODEL}} Only', both_wrong:'Both Wrong'}};

    function getComboCategory(blVal, modelVal) {{
        if(blVal===1 && modelVal===1) return 'both_correct';
        if(blVal===1 && modelVal===0) return 'bl_only';
        if(blVal===0 && modelVal===1) return 'model_only';
        return 'both_wrong';
    }}

    let comboHtml = '';
    comboModels.forEach(m => {{
        // Count each category
        const counts = {{both_correct:0, bl_only:0, model_only:0, both_wrong:0}};
        window._allCaseItems.forEach(item => {{
            if(item.bl >= 0 && item[m.key] >= 0) {{
                counts[getComboCategory(item.bl, item[m.key])]++;
            }}
        }});
        const total = Object.values(counts).reduce((a,b)=>a+b, 0);

        comboHtml += `<div class="combo-filter-group" data-model="${{m.key}}">`;
        comboHtml += `<span class="combo-label" style="color:${{m.color}}">${{m.label}} vs BL:</span>`;
        comboHtml += `<button class="combo-tab combo-all active" data-model="${{m.key}}" data-combo="all" onclick="setComboFilter('${{m.key}}','all',this)">All <span class="tab-count">${{total}}</span></button>`;
        comboCats.forEach(cat => {{
            const label = comboCatLabels[cat].replace('{{MODEL}}', m.label.split(' ')[0]);
            comboHtml += `<button class="combo-tab ${{cat}}" data-model="${{m.key}}" data-combo="${{cat}}" onclick="setComboFilter('${{m.key}}','${{cat}}',this)">${{label}} <span class="tab-count">${{counts[cat]}}</span></button>`;
        }});
        comboHtml += `</div>`;
    }});
    document.getElementById('combo-filter-section').innerHTML = comboHtml;

    // Store getComboCategory globally
    window._getComboCategory = getComboCategory;

    // Quick filter buttons for annotated cases
    const cat1Count = window._allCaseItems.filter(item => item.q.annotations && item.q.annotations.category === 'cat1').length;
    const cat2Count = window._allCaseItems.filter(item => item.q.annotations && item.q.annotations.category === 'cat2').length;
    const cat3Count = window._allCaseItems.filter(item => item.q.annotations && item.q.annotations.category === 'cat3').length;
    let qfHtml = `<span class="case-filter-label">Key Cases:</span>`;
    qfHtml += `<button class="quick-filter-btn cat1" onclick="quickFilter('cat1',this)">Baseline&#10003; All-Tool&#10007; (${{cat1Count}})</button>`;
    qfHtml += `<button class="quick-filter-btn cat2" onclick="quickFilter('cat2',this)">Baseline&#10003; 2-Tool&#10007; (${{cat2Count}})</button>`;
    qfHtml += `<button class="quick-filter-btn cat3" onclick="quickFilter('cat3',this)">Baseline&#10007; All-Tool&#10003; (${{cat3Count}})</button>`;
    document.getElementById('quick-filter-bar').innerHTML = qfHtml;

    // Correctness dropdown filters
    let corrHtml = `<span class="case-filter-label">自定义:</span>`;
    const corrModels = [
        {{key:'bl',label:'Baseline',color:'#FF9800'}},
        {{key:'gpt',label:'GPT-5.4',color:'#10a37f'}},
        {{key:'claude',label:'Claude Opus',color:'#c96442'}},
        {{key:'gem',label:'Gemini',color:'#4285f4'}},
    ];
    corrModels.forEach(m => {{
        corrHtml += `<div class="corr-filter-item"><span class="corr-model-label" style="color:${{m.color}}">${{m.label}}</span><select id="corr-${{m.key}}" onchange="onDropdownChange()"><option value="all">All</option><option value="correct">Correct</option><option value="wrong">Wrong</option></select></div>`;
    }});
    document.getElementById('corr-filter-row').innerHTML = corrHtml;

    applyAllFilters();
}}

// ── Current filter state ──
let _caseFilterType = 'cat'; // 'cat' or 'bench'
let _caseFilterVal = 'all';
let _comboModel = null; // 'gpt','claude','gem' or null
let _comboCategory = 'all'; // 'both_correct','bl_only','model_only','both_wrong','all'
let _filterMode = 'combo'; // 'combo', 'dropdown', or 'quick'
let _quickCategory = null; // 'cat1' or 'cat3'

function setCaseFilter(type, val, btn) {{
    _caseFilterType = type;
    _caseFilterVal = val;
    document.querySelectorAll('#case-filter .case-filter-btn').forEach(b => b.classList.remove('active'));
    if(btn) btn.classList.add('active');
    applyAllFilters();
}}

function setComboFilter(model, cat, btn) {{
    _filterMode = 'combo';
    _quickCategory = null;
    document.querySelectorAll('.quick-filter-btn').forEach(b => b.classList.remove('active'));
    // Reset all combo tabs in all groups to inactive
    document.querySelectorAll('.combo-tab').forEach(b => b.classList.remove('active'));
    // Toggle: if clicking same selection → reset
    if(_comboModel === model && _comboCategory === cat && cat !== 'all') {{
        _comboModel = null;
        _comboCategory = 'all';
        document.querySelectorAll('.combo-tab.combo-all').forEach(b => b.classList.add('active'));
    }} else {{
        _comboModel = (cat === 'all') ? null : model;
        _comboCategory = cat;
        // Highlight All for other groups, selected for this group
        document.querySelectorAll('.combo-tab.combo-all').forEach(b => {{
            if(b.dataset.model !== model) b.classList.add('active');
        }});
        btn.classList.add('active');
    }}
    // Reset dropdowns
    ['bl','gpt','claude','gem'].forEach(k => {{
        const sel = document.getElementById('corr-'+k);
        if(sel) {{ sel.value = 'all'; sel.className = ''; }}
    }});
    applyAllFilters();
}}

function onDropdownChange() {{
    _filterMode = 'dropdown';
    _quickCategory = null;
    _comboModel = null;
    _comboCategory = 'all';
    document.querySelectorAll('.combo-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.combo-tab.combo-all').forEach(b => b.classList.add('active'));
    document.querySelectorAll('.quick-filter-btn').forEach(b => b.classList.remove('active'));
    applyAllFilters();
}}

function quickFilter(cat, btn) {{
    // Toggle
    if(_filterMode === 'quick' && _quickCategory === cat) {{
        _filterMode = 'combo';
        _quickCategory = null;
        btn.classList.remove('active');
    }} else {{
        _filterMode = 'quick';
        _quickCategory = cat;
        // Reset other filters UI
        document.querySelectorAll('.quick-filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.combo-tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.combo-tab.combo-all').forEach(b => b.classList.add('active'));
        _comboModel = null;
        _comboCategory = 'all';
        ['bl','gpt','claude','gem'].forEach(k => {{
            const sel = document.getElementById('corr-'+k);
            if(sel) {{ sel.value = 'all'; sel.className = ''; }}
        }});
    }}
    applyAllFilters();
}}

function _passesCorrectness(item, corrFilters, corrKeys) {{
    // Quick filter (key cases)
    if(_filterMode === 'quick' && _quickCategory) {{
        const ann = item.q.annotations;
        if(!ann || ann.category !== _quickCategory) return false;
    }}
    // Combo filter
    if(_filterMode === 'combo' && _comboModel && _comboCategory !== 'all') {{
        const blVal = item.bl, mVal = item[_comboModel];
        if(blVal < 0 || mVal < 0) return false;
        if(window._getComboCategory(blVal, mVal) !== _comboCategory) return false;
    }}
    // Dropdown filter
    if(_filterMode === 'dropdown') {{
        for(const k of corrKeys) {{
            const state = corrFilters[k];
            if(state === 'all') continue;
            const val = item[k];
            if(val === -1) continue;
            if(state === 'correct' && val !== 1) return false;
            if(state === 'wrong' && val !== 0) return false;
        }}
    }}
    return true;
}}

function _passesSearch(item, searchRaw) {{
    if(!searchRaw) return true;
    const numMatch = searchRaw.match(/^#?(\\d+)$/);
    if(numMatch) return item.q.sequential_id === parseInt(numMatch[1]);
    return item.searchText.includes(searchRaw);
}}

function applyAllFilters() {{
    const searchRaw = (document.getElementById('case-search').value||'').trim().toLowerCase();

    // Dropdown filters
    const corrKeys = ['bl','gpt','claude','gem'];
    const corrFilters = {{}};
    corrKeys.forEach(k => {{
        const sel = document.getElementById('corr-'+k);
        if(sel) {{
            corrFilters[k] = sel.value;
            sel.className = sel.value==='correct'?'filter-correct':sel.value==='wrong'?'filter-wrong':'';
        }}
    }});

    // Filter
    const filtered = window._allCaseItems.filter(item => {{
        if(_caseFilterType === 'cat' && _caseFilterVal !== 'all') {{
            if(item.cat !== _caseFilterVal) return false;
        }}
        if(_caseFilterType === 'bench') {{
            if(item.bench !== _caseFilterVal) return false;
        }}
        if(!_passesCorrectness(item, corrFilters, corrKeys)) return false;
        if(!_passesSearch(item, searchRaw)) return false;
        return true;
    }});

    caseFilteredIds = filtered;
    const totalQ = window._allCaseItems.length;

    // Update total count
    if(filtered.length === totalQ) {{
        document.getElementById('cases-count').textContent = `${{totalQ}} Questions`;
    }} else {{
        document.getElementById('cases-count').textContent = `${{filtered.length}} / ${{totalQ}} Questions`;
    }}

    // ── Update category/benchmark tab counts ──
    // Count items that pass correctness + search filters (ignoring cat/bench filter)
    const catCounts = {{}};
    const benchCounts = {{}};
    let allCount = 0;
    window._allCaseItems.forEach(item => {{
        if(!_passesCorrectness(item, corrFilters, corrKeys)) return;
        if(!_passesSearch(item, searchRaw)) return;
        allCount++;
        catCounts[item.cat] = (catCounts[item.cat]||0) + 1;
        benchCounts[item.bench] = (benchCounts[item.bench]||0) + 1;
    }});
    document.querySelectorAll('#case-filter .case-filter-btn').forEach(btn => {{
        const type = btn.dataset.type;
        const val = btn.dataset.val;
        let count = allCount;
        if(type === 'cat' && val !== 'all') count = catCounts[val] || 0;
        if(type === 'bench') count = benchCounts[val] || 0;
        // Update text: keep label, update count in parens
        const text = btn.textContent.replace(/\\s*\\(\\d+\\)/, '');
        btn.textContent = `${{text}} (${{count}})`;
    }});

    // ── Update combo tab counts ──
    // Count items that pass cat/bench + search filters (ignoring combo/dropdown)
    const comboBase = window._allCaseItems.filter(item => {{
        if(_caseFilterType === 'cat' && _caseFilterVal !== 'all' && item.cat !== _caseFilterVal) return false;
        if(_caseFilterType === 'bench' && item.bench !== _caseFilterVal) return false;
        if(!_passesSearch(item, searchRaw)) return false;
        return true;
    }});
    const comboModels = ['gpt','claude','gem'];
    const comboCats = ['both_correct','bl_only','model_only','both_wrong'];
    comboModels.forEach(mk => {{
        const counts = {{all:0, both_correct:0, bl_only:0, model_only:0, both_wrong:0}};
        comboBase.forEach(item => {{
            if(item.bl >= 0 && item[mk] >= 0) {{
                const cat = window._getComboCategory(item.bl, item[mk]);
                counts[cat]++;
                counts.all++;
            }}
        }});
        document.querySelectorAll(`.combo-tab[data-model="${{mk}}"]`).forEach(btn => {{
            const combo = btn.dataset.combo;
            const span = btn.querySelector('.tab-count');
            if(span) span.textContent = counts[combo] || 0;
        }});
    }});

    // Reset to page 1
    casePage = 1;
    renderCasePage();
}}

function renderCasePage() {{
    const total = caseFilteredIds.length;
    const totalPages = Math.max(1, Math.ceil(total / casePageSize));
    if(casePage > totalPages) casePage = totalPages;
    const start = (casePage - 1) * casePageSize;
    const end = Math.min(start + casePageSize, total);
    const pageItems = caseFilteredIds.slice(start, end);

    const models = TRAJ_DATA.metadata.models;
    let html = '';
    pageItems.forEach(item => {{
        const q = item.q;
        const a = item.a;
        const color = CATEGORY_COLORS[a.ability_category]||'#888';
        const qId = `q-${{q.sequential_id}}`;
        html += `<div class="question-card" id="${{qId}}">`;
        html += `<div class="question-card-header"><div class="question-info">`;
        html += `<span class="question-seq">#${{q.sequential_id}}</span>`;
        const vdur = q.video_duration||0;
        const vmin = Math.floor(vdur/60), vsec = Math.round(vdur%60);
        html += `<span class="vid-dur">${{vmin>0?vmin+'m':''}}${{vsec}}s</span>`;
        html += `<span class="question-benchmark">${{q.benchmark}}</span>`;
        html += `<span class="question-id">Q:${{q.question_id}}</span>`;
        html += `<span class="ability-tag" style="background:${{color}}15;color:${{color}};cursor:pointer" onclick="jumpTo('ability-${{a.ability_id}}')">${{a.ability_id}}. ${{a.ability_name}}</span>`;
        html += `<span class="answer-detail">GT:${{q.ground_truth}}</span>`;
        if(q.youtube_url) html += `<a class="video-btn youtube-btn" href="${{q.youtube_url}}" target="_blank" rel="noopener" onclick="event.stopPropagation()">&#9654; YouTube</a>`;
        html += `</div><div class="question-text">${{escHtml(q.question_text)}}</div></div>`;

        // Baseline row
        if(q.baseline) {{
            const blC = q.baseline.correct;
            let blBody = '';
            if(q.baseline.reasoning) {{
                blBody += `<div class="baseline-reasoning"><div class="baseline-section-label">Reasoning</div><div class="baseline-md">${{renderMarkdown(q.baseline.reasoning)}}</div></div>`;
                blBody += `<div class="baseline-response"><div class="baseline-section-label">Response</div><div class="baseline-md">${{renderMarkdown(q.baseline.response||'')}}</div></div>`;
            }} else {{
                blBody += `<div class="baseline-response"><div class="baseline-md">${{renderMarkdown(q.baseline.response||'')}}</div></div>`;
            }}
            // Baseline annotation note
            const blAnn = q.annotations && q.annotations.baseline_note;
            if(blAnn) {{
                blBody = `<div class="ann-bl-note" style="margin:0.5rem 0.8rem">${{escHtml(blAnn)}}</div>` + blBody;
            }}
            html += `<div class="baseline-row expanded">
                <div class="baseline-row-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <span>Baseline</span>
                    <span class="correct-badge ${{blC?'correct':'wrong'}}">${{blC?'\\u2713':'\\u2717'}} ${{q.baseline.answer||'—'}}</span>
                    <span style="font-size:0.6rem;color:var(--color-text-muted)">GT: ${{q.ground_truth}}</span>
                    <span class="baseline-toggle">&#9660;</span>
                </div>
                <div class="baseline-body">${{blBody}}</div>
            </div>`;
        }}

        // Desktop: 3-column
        html += `<div class="model-compare">`;
        models.forEach(m => {{
            const traj = (q.trajectories||{{}})[m];
            const info = (q.per_model||{{}})[m]||{{}};
            const isCorrect = info.correct;
            const css = MODEL_CSS[m]||'gemini';
            html += `<div class="model-column"><div class="model-col-header ${{css}}"><span class="model-dot ${{css}}"></span>${{MODEL_SHORT[m]||m}}`;
            if(info.agent_answer !== undefined) html += `<span class="correct-badge ${{isCorrect?'correct':'wrong'}}">${{isCorrect?'\\u2713':'\\u2717'}} ${{info.agent_answer}}</span>`;
            html += `<span class="step-count">${{traj&&traj.steps? traj.steps.length+' steps':'-'}}</span></div>`;
            const stepAnns = (q.annotations&&q.annotations.step_annotations&&q.annotations.step_annotations[m])||{{}};
            // Model-level annotation note
            const modelNote = q.annotations&&q.annotations.model_notes&&q.annotations.model_notes[m];
            if(modelNote) html += `<div class="ann-model-note">${{escHtml(modelNote)}}</div>`;
            html += renderSteps(traj,m,stepAnns);
            html += `</div>`;
        }});
        html += `</div>`;

        // Mobile: tabs
        html += `<div class="model-tabs-wrapper"><div class="model-tabs">`;
        models.forEach((m,i) => {{
            html += `<div class="model-tab ${{i===0?'active':''}}" data-qid="${{qId}}" data-idx="${{i}}" onclick="switchTab(this)" style="${{i===0?'border-bottom-color:'+MODEL_COLORS[m]+';color:'+MODEL_COLORS[m]:''}}""><span class="model-dot ${{MODEL_CSS[m]||'gemini'}}"></span>${{MODEL_SHORT[m]||m}}</div>`;
        }});
        html += `</div>`;
        models.forEach((m,i) => {{
            const traj = (q.trajectories||{{}})[m];
            const stepAnns = (q.annotations&&q.annotations.step_annotations&&q.annotations.step_annotations[m])||{{}};
            html += `<div class="model-tab-panel ${{i===0?'active':''}}" data-qid="${{qId}}" data-idx="${{i}}">${{renderSteps(traj,m,stepAnns)}}</div>`;
        }});
        html += `</div>`;

        // ── Annotation panel (key cases only) ──
        if(q.annotations) {{
            const a = q.annotations;
            html += `<div class="ann-panel">`;
            html += `<div class="ann-panel-header" onclick="this.parentElement.classList.toggle('collapsed')"><span class="ann-panel-tag ${{a.category}}">${{a.tag}}</span><span class="ann-panel-title">Case Analysis</span><span class="ann-panel-toggle">click to expand/collapse</span></div>`;
            html += `<div class="ann-panel-body">`;
            html += `<div class="ann-section"><div class="ann-section-label">Summary</div><div class="ann-summary">${{escHtml(a.summary)}}</div></div>`;
            if(a.failure_reason) html += `<div class="ann-failure">${{escHtml(a.failure_reason)}}</div>`;
            if(a.baseline_note) html += `<div class="ann-section"><div class="ann-section-label">Baseline</div><div class="ann-bl-note">${{escHtml(a.baseline_note)}}</div></div>`;
            html += `</div></div>`;
        }}

        html += `</div>`;
    }});

    document.getElementById('cases-container').innerHTML = html;

    // Pagination controls
    const pagHtml = buildPaginationHtml(total, totalPages);
    document.getElementById('pagination-top').innerHTML = pagHtml;
    document.getElementById('pagination-bottom').innerHTML = pagHtml;
}}

function buildPaginationHtml(total, totalPages) {{
    if(total === 0) return `<span class="page-info">无匹配结果</span>`;
    const start = (casePage-1)*casePageSize+1;
    const end = Math.min(casePage*casePageSize, total);
    let h = `<span class="page-info">显示 ${{start}}-${{end}} / ${{total}}</span>`;
    h += `<button class="page-btn ${{casePage<=1?'disabled':''}}" onclick="goPage(${{casePage-1}})">&#8592;</button>`;
    // Show page buttons with ellipsis
    const pages = [];
    for(let i=1;i<=totalPages;i++) {{
        if(i===1||i===totalPages||Math.abs(i-casePage)<=2) pages.push(i);
        else if(pages[pages.length-1]!=='...') pages.push('...');
    }}
    pages.forEach(p => {{
        if(p==='...') {{ h += `<span class="page-info">…</span>`; }}
        else {{ h += `<button class="page-btn ${{p===casePage?'active':''}}" onclick="goPage(${{p}})">${{p}}</button>`; }}
    }});
    h += `<button class="page-btn ${{casePage>=totalPages?'disabled':''}}" onclick="goPage(${{casePage+1}})">&#8594;</button>`;
    h += `<span style="margin-left:0.5rem;font-size:0.68rem;color:var(--color-text-muted)">跳转</span>`;
    h += `<input type="number" class="page-jump-input" min="1" max="${{totalPages}}" value="${{casePage}}" onkeydown="if(event.key==='Enter')goPage(parseInt(this.value))" style="width:3.5rem;padding:0.2rem 0.3rem;border:1px solid var(--color-border);border-radius:4px;font-size:0.68rem;text-align:center;font-family:var(--font-mono)">`;
    h += `<span style="font-size:0.62rem;color:var(--color-text-muted)">/ ${{totalPages}}</span>`;
    h += `<select class="page-size-select" onchange="casePageSize=parseInt(this.value);casePage=1;renderCasePage()">`;
    [1,5,10,20].forEach(n => {{ h += `<option value="${{n}}" ${{n===casePageSize?'selected':''}}>${{n}}/页</option>`; }});
    h += `</select>`;
    return h;
}}

function goPage(p) {{
    const totalPages = Math.max(1, Math.ceil(caseFilteredIds.length / casePageSize));
    if(p<1||p>totalPages) return;
    casePage = p;
    renderCasePage();
    document.getElementById('sec-cases').scrollIntoView({{behavior:'smooth'}});
}}

// Keyboard navigation: left/right arrow to switch pages
document.addEventListener('keydown', function(e) {{
    // Skip if user is typing in an input/select/textarea
    const tag = (e.target.tagName||'').toLowerCase();
    if(tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if(e.key === 'ArrowLeft') {{ e.preventDefault(); goPage(casePage - 1); }}
    else if(e.key === 'ArrowRight') {{ e.preventDefault(); goPage(casePage + 1); }}
}});

function renderSteps(traj, model, stepAnns) {{
    if(!traj||!traj.steps||!traj.steps.length) return '<div class="no-tools">--</div>';
    const uid = 'st'+Math.random().toString(36).slice(2,8);
    const anns = stepAnns || {{}};
    let html = '<div class="step-list">';
    traj.steps.forEach((s,si) => {{
        const isR = !s.tool||s.tool==='null';
        const label = isR?'Reasoning / Final Answer':s.tool;
        const detailId = `${{uid}}-${{si}}`;
        const hasIO = s.args_brief || s.output_brief;
        html += `<div class="tool-step"><div class="tool-step-header"><span class="step-num ${{isR?'reasoning':''}}">${{s.step}}</span>`;
        if(isR) {{
            html += `<span class="step-name reasoning">${{label}}</span>`;
        }} else {{
            html += `<span class="step-name" onclick="jumpTo('tool-${{s.tool}}')">${{label}}</span>`;
        }}
        html += `</div>`;
        // Thinking (collapsible, above content)
        if(s.thinking) {{
            html += `<div class="step-collapsible thinking"><div class="step-collapsible-header" onclick="this.parentElement.classList.toggle('expanded')"><span class="step-collapsible-toggle">&#9654;</span>Thinking</div><div class="step-collapsible-body">${{escHtml(s.thinking)}}</div></div>`;
        }}
        // Content (purpose)
        if(s.purpose) html += `<div class="step-purpose">${{renderMarkdown(s.purpose)}}</div>`;
        // Params bar (collapsible, below content)
        if(hasIO) {{
            let paramsBody = '';
            if(s.args_brief) paramsBody += `<div class="step-block input"><span class="step-block-label">Input</span><div class="step-block-content">${{renderParamsTable(s.args_brief)}}</div></div>`;
            if(s.output_brief) paramsBody += `<div class="step-block output"><span class="step-block-label">Output</span><div class="step-block-content">${{renderParamsTable(s.output_brief)}}</div></div>`;
            html += `<div class="step-collapsible params"><div class="step-collapsible-header" onclick="this.parentElement.classList.toggle('expanded')"><span class="step-collapsible-toggle">&#9654;</span>Params</div><div class="step-collapsible-body">${{paramsBody}}</div></div>`;
        }}
        // Embedded images (key cases only)
        if(s.images && s.images.length) {{
            html += `<div class="step-images">`;
            s.images.forEach(img => {{
                html += `<div class="step-img-thumb" onclick="showLightbox(this)"><img src="${{img.src}}" loading="lazy" alt="${{escHtml(img.label)}}"><span class="step-img-label">${{escHtml(img.label)}}</span></div>`;
            }});
            html += `</div>`;
        }}
        // Step annotation
        const ann = anns[String(s.step)];
        if(ann) {{
            html += `<div class="step-ann">${{escHtml(ann)}}</div>`;
        }}
        html += `</div>`;
    }});
    html += '</div>'; return html;
}}


function toggleSection(name) {{
    const header = document.querySelector(`#sec-${{name}} .collapsible-header`);
    const body = document.getElementById(name+'-body');
    if(header) header.classList.toggle('collapsed');
    if(body) body.classList.toggle('collapsed');
}}

function toggleExecDetail(id) {{
    const el = document.getElementById(id);
    if(el) el.classList.toggle('expanded');
}}

function tryParseJson(str) {{
    if(!str) return null;
    let s = str.trim();
    const mdMatch = s.match(/^```(?:json)?\\s*\\n?([\\s\\S]*?)\\n?\\s*```$/) || s.match(/^```(?:json)?\\s*\\n?([\\s\\S]+)$/);
    if(mdMatch) s = mdMatch[1].trim();
    try {{ return JSON.parse(s); }} catch(e) {{}}
    return null;
}}

function renderParamValue(val) {{
    if(val===null||val===undefined) return '<span style="color:var(--color-text-muted)">null</span>';
    if(typeof val==='string') return escHtml(val);
    if(Array.isArray(val)) {{
        if(val.length===0) return '[]';
        const allObj = val.every(item => typeof item==='object'&&item!==null&&!Array.isArray(item));
        if(allObj) return renderArrayTable(val);
        return escHtml(JSON.stringify(val));
    }}
    if(typeof val==='object') {{
        const entries = Object.entries(val);
        if(entries.length===0) return '{{}}';
        const allPrim = entries.every(([,v]) => typeof v!=='object'||v===null);
        if(allPrim && entries.length<=4) return escHtml(JSON.stringify(val));
        return renderObjTable(val);
    }}
    return escHtml(String(val));
}}

function renderObjTable(obj) {{
    const entries = Object.entries(obj);
    if(!entries.length) return '';
    let html = '<table class="param-table"><tbody>';
    entries.forEach(([key,val]) => {{
        html += `<tr><td class="param-key">${{escHtml(key)}}</td><td class="param-val">${{renderParamValue(val)}}</td></tr>`;
    }});
    html += '</tbody></table>';
    return html;
}}

function renderArrayTable(arr) {{
    if(!arr.length) return '';
    const allObj = arr.every(item => typeof item==='object'&&item!==null&&!Array.isArray(item));
    if(allObj) {{
        const allKeys = [...new Set(arr.flatMap(item => Object.keys(item)))];
        let html = '<table class="param-table"><thead><tr>';
        allKeys.forEach(k => {{ html += `<th class="param-key">${{escHtml(k)}}</th>`; }});
        html += '</tr></thead><tbody>';
        arr.forEach(item => {{
            html += '<tr>';
            allKeys.forEach(k => {{
                const v = item[k];
                html += `<td class="param-val">${{v!==undefined?renderParamValue(v):''}}</td>`;
            }});
            html += '</tr>';
        }});
        html += '</tbody></table>';
        return html;
    }}
    let html = '<table class="param-table"><tbody>';
    arr.forEach((item,i) => {{
        html += `<tr><td class="param-key">[${{i}}]</td><td class="param-val">${{renderParamValue(item)}}</td></tr>`;
    }});
    html += '</tbody></table>';
    return html;
}}

function renderParamsTable(str) {{
    if(!str) return '';
    let parsed = tryParseJson(str);
    if(parsed!==null) {{
        if(typeof parsed==='string') {{ const inner=tryParseJson(parsed); if(inner!==null) parsed=inner; else return escHtml(parsed); }}
        if(typeof parsed!=='object'||parsed===null) return escHtml(String(parsed));
        if(Array.isArray(parsed)) return renderArrayTable(parsed);
        return renderObjTable(parsed);
    }}
    return `<span style="white-space:pre-wrap">${{escHtml(str)}}</span>`;
}}

function renderBenchStats() {{
    const benchOrder = ['lvbench','longvideobench','videomme'];
    const benchLabels = {{'lvbench':'LVBench','longvideobench':'LongVideoBench','videomme':'Video-MME'}};
    const modelOrder = ['baseline','gpt-5.4','claude-opus-4-6','gemini'];
    const modelLabels = {{'baseline':'Baseline (Gemini 3.1 Pro Direct)','gpt-5.4':'GPT-5.4','claude-opus-4-6':'Claude Opus 4.6','gemini':'Gemini 3.0 Pro'}};
    const modelShort = {{'baseline':'Baseline','gpt-5.4':'GPT-5.4','claude-opus-4-6':'Claude','gemini':'Gemini'}};
    const modelColors = {{'baseline':'#FF9800','gpt-5.4':'#10a37f','claude-opus-4-6':'#c96442','gemini':'#4285f4'}};
    const CPT = 4;

    const fk = v => v >= 1000 ? (v/1000).toFixed(1)+'k' : v < 1 ? '<1' : Math.round(v).toString();
    const fmtDur = d => {{ const m=Math.floor(d/60),s=Math.round(d%60); return m>0 ? m+'m'+s+'s' : s+'s'; }};

    // ── Compute accuracy per model per benchmark + overall average ──
    const accData = {{}};
    modelOrder.forEach(mk => {{
        let totalCorrect = 0, totalN = 0;
        benchOrder.forEach(ds => {{
            const s = BENCH_STATS[ds] && BENCH_STATS[ds][mk];
            const acc = s ? s.accuracy : null;
            if(!accData[mk]) accData[mk] = {{}};
            accData[mk][ds] = acc;
            if(s) {{ totalCorrect += Math.round(s.accuracy * s.n / 100); totalN += s.n; }}
        }});
        // Overall = total correct / total questions (not avg of per-benchmark accuracies)
        accData[mk].avg = totalN > 0 ? Math.round(totalCorrect / totalN * 1000) / 10 : null;
    }});

    // Find max accuracy per column
    const maxAcc = {{}};
    [...benchOrder, 'avg'].forEach(col => {{
        let best = -1;
        modelOrder.forEach(mk => {{
            const v = col === 'avg' ? accData[mk].avg : accData[mk][col];
            if(v !== null && v > best) best = v;
        }});
        maxAcc[col] = best;
    }});

    // Baseline accuracy for delta
    const blAcc = accData['baseline'] || {{}};

    // ── Table 1: 准确率对比 ──
    let html = `<div style="margin-bottom:1.5rem">`;
    html += `<div style="font-size:0.78rem;font-weight:700;margin-bottom:0.5rem;color:var(--color-text)">准确率对比</div>`;
    html += `<table class="model-compare-table" style="font-size:0.75rem;width:auto">`;
    html += `<thead><tr><th style="min-width:180px">模型</th>`;
    benchOrder.forEach(ds => {{
        const bl = BENCH_STATS[ds] && BENCH_STATS[ds].baseline;
        const n = bl ? bl.n : '?';
        const dur = bl ? fmtDur(bl.avg_vid_duration) : '?';
        html += `<th style="text-align:center;min-width:100px">${{benchLabels[ds]}}<br><span style="font-weight:400;font-size:0.58rem;color:var(--color-text-muted)">${{n}}题 · 均 ${{dur}}</span></th>`;
    }});
    html += `<th style="text-align:center;min-width:80px;background:#f8f9fa">总体<br><span style="font-weight:400;font-size:0.58rem;color:var(--color-text-muted)">116题</span></th>`;
    html += `</tr></thead><tbody>`;

    modelOrder.forEach(mk => {{
        const color = modelColors[mk];
        const isBl = mk === 'baseline';
        html += `<tr${{isBl?' style="background:#fff7ed"':''}}>`;
        html += `<td style="white-space:nowrap"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{color}};margin-right:0.4rem;vertical-align:middle"></span>${{modelLabels[mk]}}</td>`;

        [...benchOrder, 'avg'].forEach(col => {{
            const acc = col === 'avg' ? accData[mk].avg : accData[mk][col];
            const isMax = acc !== null && acc === maxAcc[col];
            const isAvgCol = col === 'avg';
            let cell = '';
            if(acc === null) {{
                cell = '—';
            }} else {{
                const accStr = isMax ? `<strong>${{acc}}%</strong>` : `${{acc}}%`;
                if(!isBl) {{
                    const blVal = col === 'avg' ? blAcc.avg : blAcc[col];
                    if(blVal !== null && blVal !== undefined) {{
                        const delta = Math.round((acc - blVal)*10)/10;
                        const sign = delta >= 0 ? '+' : '';
                        const dColor = delta > 0 ? '#16a34a' : delta < 0 ? '#dc2626' : '#888';
                        cell = `${{accStr}} <span style="font-size:0.6rem;color:${{dColor}}">${{sign}}${{delta}}</span>`;
                    }} else {{
                        cell = accStr;
                    }}
                }} else {{
                    cell = accStr;
                }}
            }}
            const bgStyle = isAvgCol ? 'background:#f8f9fa;' : '';
            html += `<td style="text-align:center;${{bgStyle}}">${{cell}}</td>`;
        }});
        html += `</tr>`;
    }});
    html += `</tbody></table></div>`;

    // ── Table 2: 上下文长度统计 ──
    html += `<div style="margin-bottom:0.8rem">`;
    html += `<div style="font-size:0.78rem;font-weight:700;margin-bottom:0.5rem;color:var(--color-text)">上下文长度统计 <span style="font-weight:400;font-size:0.62rem;color:var(--color-text-muted)">（单位：k tokens）</span></div>`;
    html += `<table class="model-compare-table" style="font-size:0.72rem">`;
    html += `<thead><tr><th style="min-width:140px">模型</th>`;
    benchOrder.forEach(ds => {{
        html += `<th colspan="4" style="text-align:center;border-left:2px solid var(--color-border)">${{benchLabels[ds]}}</th>`;
    }});
    html += `</tr><tr><th></th>`;
    benchOrder.forEach(() => {{
        html += `<th style="border-left:2px solid var(--color-border);font-size:0.6rem">输入上下文</th><th style="font-size:0.6rem">输出</th><th style="font-size:0.6rem">工具输入</th><th style="font-size:0.6rem">工具输出</th>`;
    }});
    html += `</tr></thead><tbody>`;

    modelOrder.forEach(mk => {{
        const color = modelColors[mk];
        const isBl = mk === 'baseline';
        html += `<tr${{isBl?' style="background:#fff7ed"':''}}>`;
        html += `<td style="white-space:nowrap"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{color}};margin-right:0.4rem;vertical-align:middle"></span><strong>${{modelShort[mk]}}</strong></td>`;

        benchOrder.forEach(ds => {{
            const s = BENCH_STATS[ds] && BENCH_STATS[ds][mk];
            if(!s) {{
                html += `<td style="border-left:2px solid var(--color-border)" colspan="4">—</td>`;
                return;
            }}
            if(isBl) {{
                html += `<td style="border-left:2px solid var(--color-border)">${{fk(s.avg_input_tokens)}}</td>`;
                html += `<td>${{fk(s.avg_output_tokens)}}</td>`;
                html += `<td style="color:#aaa">—</td><td style="color:#aaa">—</td>`;
            }} else {{
                const ctxTok = s.avg_final_input_chars ? Math.round(s.avg_final_input_chars / CPT) : 0;
                const outTok = s.avg_last_output_chars ? Math.round(s.avg_last_output_chars / CPT) : 0;
                const tiTok = s.avg_tool_input_chars ? Math.round(s.avg_tool_input_chars / CPT) : 0;
                const toTok = s.avg_tool_output_chars ? Math.round(s.avg_tool_output_chars / CPT) : 0;
                html += `<td style="border-left:2px solid var(--color-border)">${{fk(ctxTok)}}</td>`;
                html += `<td>${{fk(outTok)}}</td>`;
                html += `<td>${{fk(tiTok)}}</td>`;
                html += `<td>${{fk(toTok)}}</td>`;
            }}
        }});
        html += `</tr>`;
    }});
    html += `</tbody></table></div>`;

    // Legend
    html += `<div style="font-size:0.62rem;color:var(--color-text-muted);line-height:1.7">`;
    html += `<strong>输入上下文</strong> = 最后一轮 AI 调用的输入长度（含 system + user + 历史 AI 输出 + 工具返回）<br>`;
    html += `<strong>工具输入</strong> = 输入上下文中工具调用参数总长度；<strong>工具输出</strong> = 输入上下文中工具返回结果总长度<br>`;
    html += `Baseline 为实际 token 数（含 image tokens 占 99%+）；Tool 模型为估算值（text chars ÷ 4，不含图像）`;
    html += `</div>`;

    document.getElementById('bench-container').innerHTML = html;
}}

function toggleToolCard(h) {{ h.parentElement.classList.toggle('expanded'); }}
function filterTools(d,v,btn) {{ document.querySelectorAll('#tool-filter .filter-btn').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); document.querySelectorAll('.tool-card').forEach(c=>{{ if(d==='all'){{c.style.display='';return;}} if(d==='type'){{c.style.display=c.dataset.type===v?'':'none';return;}} if(d==='level'){{c.style.display=c.dataset.level===v?'':'none';return;}} }}); }}
function switchTab(tabEl) {{
    const qid=tabEl.dataset.qid, idx=tabEl.dataset.idx;
    document.querySelectorAll(`.model-tab[data-qid="${{qid}}"]`).forEach(t=>{{t.classList.remove('active');t.style.borderBottomColor='transparent';t.style.color='var(--color-text-muted)';}});
    document.querySelectorAll(`.model-tab-panel[data-qid="${{qid}}"]`).forEach(p=>p.classList.remove('active'));
    tabEl.classList.add('active');
    const m = TRAJ_DATA.metadata.models[idx];
    tabEl.style.borderBottomColor=MODEL_COLORS[m]; tabEl.style.color=MODEL_COLORS[m];
    document.querySelector(`.model-tab-panel[data-qid="${{qid}}"][data-idx="${{idx}}"]`).classList.add('active');
}}
function renderMarkdown(s) {{ if(typeof marked!=='undefined'&&marked.parse) return marked.parse(s||''); return escHtml(s||'').replace(/\\n/g,'<br>'); }}

function showLightbox(thumb) {{
    const img = thumb.querySelector('img');
    if(!img) return;
    const overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';
    overlay.innerHTML = `<img src="${{img.src}}">`;
    overlay.onclick = () => overlay.remove();
    document.addEventListener('keydown', function handler(e) {{
        if(e.key==='Escape') {{ overlay.remove(); document.removeEventListener('keydown',handler); }}
    }});
    document.body.appendChild(overlay);
}}
function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>'''
    return html


if __name__ == "__main__":
    main()
