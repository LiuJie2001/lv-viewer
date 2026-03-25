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
from collections import Counter, defaultdict

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


def parse_trajectory_steps(trajectory):
    """Parse trajectory messages into steps with tool args and output."""
    steps = []
    step_num = 0
    # Build tool_call_id → output map for matching
    pending_tool_calls = []  # list of (step_index, tool_call_name)
    tool_output_queue = []

    # First pass: pair AI tool_calls with subsequent tool outputs
    paired = []  # list of (ai_content, [(tool_name, args, output), ...])
    i = 0
    while i < len(trajectory):
        item = trajectory[i]
        if item.get("role") == "ai":
            content = (item.get("content") or "")
            tool_calls = item.get("tool_calls") or []
            if tool_calls:
                tools_with_output = []
                for tc in tool_calls:
                    # Look ahead for matching tool output
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
                paired.append(("tool_call", content, tools_with_output))
                # Skip past the tool outputs we consumed
                i += 1
                while i < len(trajectory) and trajectory[i].get("role") == "tool":
                    i += 1
                continue
            else:
                paired.append(("reasoning", content, []))
        i += 1

    # Build steps
    for entry_type, content, tools_data in paired:
        if entry_type == "tool_call":
            for tool_name, args, output in tools_data:
                step_num += 1
                args_brief = json.dumps(args, ensure_ascii=False) if args else ""
                output_brief = output if output else ""
                steps.append({
                    "step": step_num,
                    "tool": tool_name,
                    "purpose": content if content else "",
                    "args_brief": args_brief,
                    "output_brief": output_brief,
                })
        else:
            step_num += 1
            steps.append({
                "step": step_num,
                "tool": None,
                "purpose": content,
            })
    return steps


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


def main():
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
                steps = parse_trajectory_steps(rec.get("trajectory", []))
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
        .model-column {{ border-right:1px solid var(--color-border); min-height:60px; }}
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
        .no-tools {{ padding:1rem; text-align:center; font-size:0.75rem; color:var(--color-text-muted); }}

        /* Tool I/O detail blocks */
        .exec-tool-item {{ display:flex; align-items:center; gap:0.35rem; padding:0.15rem 0 0.15rem 1.6rem; }}
        .exec-tool-dot {{ width:6px; height:6px; border-radius:50%; background:var(--color-accent); flex-shrink:0; }}
        .exec-tool-detail {{ display:none; margin:0.15rem 0 0.3rem 1.6rem; }}
        .exec-tool-detail.expanded {{ display:block; }}
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
let casePageSize = 10;
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

    // Correctness dropdown filters
    let corrHtml = `<span class="case-filter-label">正误筛选:</span>`;
    const corrModels = [
        {{key:'bl',label:'Baseline',color:'#FF9800'}},
        {{key:'gpt',label:'GPT-5.4',color:'#10a37f'}},
        {{key:'claude',label:'Claude Opus',color:'#c96442'}},
        {{key:'gem',label:'Gemini',color:'#4285f4'}},
    ];
    corrModels.forEach(m => {{
        corrHtml += `<div class="corr-filter-item"><span class="corr-model-label" style="color:${{m.color}}">${{m.label}}</span><select id="corr-${{m.key}}" onchange="applyAllFilters()"><option value="all">All</option><option value="correct">✓ 正确</option><option value="wrong">✗ 错误</option></select></div>`;
    }});
    document.getElementById('corr-filter-row').innerHTML = corrHtml;

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

    applyAllFilters();
}}

// ── Current filter state ──
let _caseFilterType = 'cat'; // 'cat' or 'bench'
let _caseFilterVal = 'all';

function setCaseFilter(type, val, btn) {{
    _caseFilterType = type;
    _caseFilterVal = val;
    document.querySelectorAll('#case-filter .case-filter-btn').forEach(b => b.classList.remove('active'));
    if(btn) btn.classList.add('active');
    applyAllFilters();
}}

function applyAllFilters() {{
    const searchRaw = (document.getElementById('case-search').value||'').trim().toLowerCase();

    // Correctness filters
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
        // Category / Benchmark filter
        if(_caseFilterType === 'cat' && _caseFilterVal !== 'all') {{
            if(item.cat !== _caseFilterVal) return false;
        }}
        if(_caseFilterType === 'bench') {{
            if(item.bench !== _caseFilterVal) return false;
        }}
        // Correctness
        for(const k of corrKeys) {{
            const state = corrFilters[k];
            if(state === 'all') continue;
            const val = item[k];
            if(val === -1) continue;
            if(state === 'correct' && val !== 1) return false;
            if(state === 'wrong' && val !== 0) return false;
        }}
        // Search
        if(searchRaw) {{
            // Support #num exact match
            const numMatch = searchRaw.match(/^#?(\\d+)$/);
            if(numMatch) {{
                if(item.q.sequential_id !== parseInt(numMatch[1])) return false;
            }} else {{
                if(!item.searchText.includes(searchRaw)) return false;
            }}
        }}
        return true;
    }});

    caseFilteredIds = filtered;
    const totalQ = window._allCaseItems.length;

    // Update count
    if(filtered.length === totalQ) {{
        document.getElementById('cases-count').textContent = `${{totalQ}} Questions`;
    }} else {{
        document.getElementById('cases-count').textContent = `${{filtered.length}} / ${{totalQ}} Questions`;
    }}

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
            html += `<div class="baseline-row expanded" onclick="this.classList.toggle('expanded')">
                <div class="baseline-row-header">
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
            html += renderSteps(traj,m);
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
            html += `<div class="model-tab-panel ${{i===0?'active':''}}" data-qid="${{qId}}" data-idx="${{i}}">${{renderSteps(traj,m)}}</div>`;
        }});
        html += `</div>`;

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
    h += `<select class="page-size-select" onchange="casePageSize=parseInt(this.value);casePage=1;renderCasePage()">`;
    [10,20,50].forEach(n => {{ h += `<option value="${{n}}" ${{n===casePageSize?'selected':''}}>${{n}}/页</option>`; }});
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

function renderSteps(traj, model) {{
    if(!traj||!traj.steps||!traj.steps.length) return '<div class="no-tools">--</div>';
    const uid = 'st'+Math.random().toString(36).slice(2,8);
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
            if(hasIO) html += `<span style="font-size:0.55rem;color:var(--color-accent);cursor:pointer;margin-left:auto" onclick="toggleExecDetail('${{detailId}}')">&#9654;</span>`;
        }}
        html += `</div>`;
        if(s.purpose) html += `<div class="step-purpose">${{escHtml(s.purpose)}}</div>`;
        if(hasIO) {{
            html += `<div class="exec-tool-detail" id="${{detailId}}">`;
            if(s.args_brief) html += `<div class="step-block input"><span class="step-block-label">Input</span><div class="step-block-content">${{renderParamsTable(s.args_brief)}}</div></div>`;
            if(s.output_brief) html += `<div class="step-block output"><span class="step-block-label">Output</span><div class="step-block-content">${{renderParamsTable(s.output_brief)}}</div></div>`;
            html += `</div>`;
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
    const modelLabels = {{'baseline':'Baseline','gpt-5.4':'GPT-5.4','claude-opus-4-6':'Claude Opus 4.6','gemini':'Gemini'}};
    const modelColors = {{'baseline':'#FF9800','gpt-5.4':'#10a37f','claude-opus-4-6':'#c96442','gemini':'#4285f4'}};
    const CPT = 4;

    const fk = v => v >= 1000 ? (v/1000).toFixed(1)+'k' : v < 1 ? '<1' : Math.round(v).toString();
    const fmtDur = d => {{ const m=Math.floor(d/60),s=Math.round(d%60); return m>0 ? m+'m'+s+'s' : s+'s'; }};

    let html = '<div style="overflow-x:auto"><table class="model-compare-table" style="font-size:0.72rem">';

    // Header row 1: benchmark groups
    html += `<thead><tr><th rowspan="2" style="min-width:140px">Model</th>`;
    benchOrder.forEach(ds => {{
        const bl = BENCH_STATS[ds] && BENCH_STATS[ds].baseline;
        const n = bl ? bl.n : '?';
        const dur = bl ? fmtDur(bl.avg_vid_duration) : '?';
        html += `<th colspan="5" style="text-align:center;border-left:2px solid var(--color-border)">${{benchLabels[ds]}}<br><span style="font-weight:400;font-size:0.6rem;color:var(--color-text-muted)">${{n}} cases · avg ${{dur}}</span></th>`;
    }});
    html += `</tr>`;

    // Header row 2: metrics
    html += `<tr>`;
    benchOrder.forEach(() => {{
        html += `<th style="border-left:2px solid var(--color-border)">Acc</th><th>Final Input</th><th>Final Output</th><th>Tool Input</th><th>Tool Output</th>`;
    }});
    html += `</tr></thead><tbody>`;

    // Model rows
    modelOrder.forEach(mk => {{
        const color = modelColors[mk];
        html += `<tr${{mk==='baseline'?' style="background:#fff7ed"':''}}>`;
        html += `<td style="white-space:nowrap"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{color}};margin-right:0.4rem;vertical-align:middle"></span><strong>${{modelLabels[mk]}}</strong></td>`;

        benchOrder.forEach(ds => {{
            const s = BENCH_STATS[ds] && BENCH_STATS[ds][mk];
            if(!s) {{
                html += `<td style="border-left:2px solid var(--color-border)" colspan="5">—</td>`;
                return;
            }}

            if(mk === 'baseline') {{
                html += `<td style="border-left:2px solid var(--color-border)"><strong>${{s.accuracy}}%</strong></td>`;
                html += `<td>${{fk(s.avg_input_tokens)}}</td>`;
                html += `<td>${{fk(s.avg_output_tokens)}}</td>`;
                html += `<td>—</td><td>—</td>`;
            }} else {{
                const ctxTok = s.avg_final_input_chars ? Math.round(s.avg_final_input_chars / CPT) : 0;
                const outTok = s.avg_last_output_chars ? Math.round(s.avg_last_output_chars / CPT) : 0;
                const tiTok = s.avg_tool_input_chars ? Math.round(s.avg_tool_input_chars / CPT) : 0;
                const toTok = s.avg_tool_output_chars ? Math.round(s.avg_tool_output_chars / CPT) : 0;
                html += `<td style="border-left:2px solid var(--color-border)"><strong>${{s.accuracy}}%</strong></td>`;
                html += `<td>${{fk(ctxTok)}}</td>`;
                html += `<td>${{fk(outTok)}}</td>`;
                html += `<td>${{fk(tiTok)}}</td>`;
                html += `<td>${{fk(toTok)}}</td>`;
            }}
        }});
        html += `</tr>`;
    }});

    html += `</tbody></table></div>`;

    // Legend
    html += `<div style="margin-top:0.8rem;font-size:0.65rem;color:var(--color-text-muted);line-height:1.8">`;
    html += `所有长度单位：<strong>k tokens</strong><br>`;
    html += `<strong>Final Input</strong> = 最后一轮 AI 调用的输入上下文长度 = system + user + 所有 AI 历史输出 + 所有 Tool Output<br>`;
    html += `<strong>Final Output</strong> = 最后一轮 AI 输出长度<br>`;
    html += `<strong>Tool Input</strong> = Final Input 中工具调用参数的总长度；<strong>Tool Output</strong> = Final Input 中工具返回结果的总长度<br>`;
    html += `Baseline 为实际 token 数，含 image tokens 占 99%+；Tool 模型为估算 token 数 text chars ÷ 4，不含图像`;
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
function escHtml(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>'''
    return html


if __name__ == "__main__":
    main()
