# Tool v12 优化分析可视化页面设计

## 目标

创建页面 `09-v12-optimization-analysis.html`，展示 tool v12 对 121 个长视频理解 case 的优化效果，聚焦三个维度：

1. **改进 Case 分析**：baseline 和 tool_v10 做错的 case，v12 的 DINO-X 检测 + thinking 是否起作用
2. **Token 消耗对比**：baseline（直接推理）vs tool use 两种方案的 token 开销
3. **视频时长标注**：每个 case 带视频时长，分析时长与难度的关系

## 数据流

### 输入数据源

| 文件 | 说明 |
|------|------|
| `121case-gemini_3.1or3.0-tool_v10orv12.jsonl` | 合并后的 tool 执行结果（含轨迹） |
| `s5_agent_results_direct_gemini-3.1-pro.jsonl` | Baseline 直接推理结果 |
| `token_comparison.json` | Token 对比统计 |
| `s4_tool_registry_v12.json` | v12 工具注册表（33 工具） |
| `s5_agent_results_tool_v12_*_trajectories/` | v12 轨迹步骤详情 |
| `bbox_vis_*/` | DINO-X 检测可视化图片 |

所有输入位于 `/root/paddlejob/workspace/env_run/output/bwh/lj/video_cases_tools/data/outputs/`。

### 预处理脚本

`scripts/prepare_v12_analysis.py` 整合所有数据源为单一 JSON。

**输出**：`data/v12-optimization-analysis.json`
**图片**：拷贝检测图片到 `data/v12-detection-images/<question_id>/`

### 预处理逻辑

**Case 来源说明**：
- 121 case 中，106 个来自 v10（gemini-3.1-pro），15 个来自 v12（gemini-3-pro）
- v10 结果中做错的 case 被加入 redo list，用 v12 重跑并替换
- `source_tool_version` 字段标记每个 case 实际使用的工具版本
- 两个版本使用不同工具注册表（v10 用 VLM 模拟检测，v12 用真实 DINO-X API）
- 准确率对比始终是：baseline(direct) vs tool(merged v10+v12)

**Category 分类规则**（与页面 08 对齐）：
- `tool_only`：baseline 错 + tool 对（= 工具改进的 case）
- `direct_only`：baseline 对 + tool 错（= 工具退步的 case）
- `both_correct`：都对
- `both_wrong`：都错

> 注：08 页面还有 `tool_failed`、`direct_failed`、`both_failed` 三种失败态。本页面中 merged 数据不存在执行失败的情况（失败的已被 v12 重跑），因此只使用四种分类。

**Token 聚合**：
- Tool token：取 JSONL 中 `token_usage.input_tokens` + `token_usage.output_tokens`（若 JSONL 无此字段，从 `token_comparison.json` 的 `per_case` 数组获取）
- Direct token：从 baseline JSONL 的 `token_usage` 获取（含 `input_tokens`、`output_tokens`、`image_tokens`、`thoughts_tokens`）

**视频时长**：
- 优先从 baseline JSONL 的 `video_info.duration` 获取（已由 ffprobe 提取）
- 若缺失，从 merged JSONL 的 `video_duration_seconds` 获取

**轨迹步骤**：
- 从 merged JSONL 的 `trajectory` 数组读取，保留原始步骤顺序（步骤编号从 0 开始，连续递增）
- AI 步骤：`role=ai`，提取 `content` 作为 thinking，`tool_calls` 作为工具调用
- Tool 步骤：`role=tool`，提取 `name` 和 `content`（输出）
- DINO-X 标记：当 tool name 为 `object_detection` 或 `instance_segmentation` 时，标记 `is_dinox=true`

**检测图片**：
- 查找 `bbox_vis_<question_id>/` 目录下的图片文件
- 将匹配的图片拷贝到 `data/v12-detection-images/<question_id>/`
- 若 question_id 无对应图片目录，`detection_images` 为空数组

**错误处理**：
- 缺失 baseline 数据：`baseline` 字段设为 `null`，category 标记为 `both_wrong`
- 缺失轨迹：`trajectory` 为空数组
- 缺失图片：`detection_images` 为空数组，不影响渲染

### 输出 JSON 结构

```json
{
  "metadata": {
    "total_cases": 121,
    "datasets": ["longvideobench", "videomme", "lvbench"],
    "generated_date": "2026-03-24",
    "models": {
      "baseline": "gemini-3.1-pro-preview",
      "tool_v10": "gemini-3.1-pro",
      "tool_v12": "gemini-3-pro"
    }
  },
  "summary": {
    "baseline_accuracy": { "total": 121, "correct": 85, "rate": 0.70 },
    "tool_accuracy": { "total": 121, "correct": 90, "rate": 0.74 },
    "by_dataset": {
      "longvideobench": { "total": 50, "baseline_correct": 35, "tool_correct": 38 },
      "videomme": { "total": 40, "baseline_correct": 30, "tool_correct": 32 },
      "lvbench": { "total": 31, "baseline_correct": 20, "tool_correct": 20 }
    },
    "categories": {
      "tool_only": 12,
      "direct_only": 7,
      "both_correct": 78,
      "both_wrong": 24
    },
    "token_stats": {
      "overall": { "tool_mean": 25000, "direct_mean": 18000, "ratio": 1.39 },
      "by_dataset": {
        "longvideobench": { "tool_mean": 28000, "direct_mean": 20000, "ratio": 1.40 },
        "videomme": { "tool_mean": 22000, "direct_mean": 16000, "ratio": 1.38 },
        "lvbench": { "tool_mean": 24000, "direct_mean": 17000, "ratio": 1.41 }
      }
    },
    "video_duration_stats": { "min": 12.5, "max": 3600.0, "mean": 245.3, "median": 180.0 },
    "duration_distribution": [
      { "label": "<1min", "count": 15 },
      { "label": "1-3min", "count": 45 },
      { "label": "3-10min", "count": 40 },
      { "label": ">10min", "count": 21 }
    ]
  },
  "tool_registry": [
    {
      "name": "object_detection",
      "processing_level": "image",
      "backend": "dinox",
      "description": "...",
      "input": {},
      "output": {},
      "implementation": { "type": "api", "backend": "dinox-api", "notes": "..." },
      "v10_diff": { "backend_change": "vlm → dinox-api" }
    }
  ],
  "cases": [
    {
      "question_id": "longvideobench_xxx_0",
      "dataset": "longvideobench",
      "question": "What object does the person pick up?",
      "options": ["A. a book", "B. a cup", "C. a phone", "D. a bag"],
      "ground_truth": "A",
      "category": "tool_only",
      "source_tool_version": "v12",
      "video_duration_seconds": 45.67,
      "video_info": { "total_frames": 1370, "fps": 30.0, "duration": 45.67 },

      "baseline": {
        "answer": "B",
        "correct": false,
        "reasoning": "The person reaches for...(markdown)",
        "response": "Based on my analysis...(markdown)",
        "token_usage": {
          "input": 12000,
          "output": 500,
          "image": 8000,
          "thoughts": 2000,
          "total": 14500
        },
        "duration_seconds": 15.3
      },

      "tool": {
        "answer": "A",
        "correct": true,
        "steps_count": 8,
        "token_usage": { "input": 20000, "output": 3800, "total": 23800 },
        "trajectory": [
          {
            "step": 0,
            "role": "ai",
            "thinking": "需要先提取关键帧来分析视频内容...",
            "tool_calls": [{ "name": "frame_extraction", "args": {"video_path": "...", "count": 16} }]
          },
          {
            "step": 1,
            "role": "tool",
            "name": "frame_extraction",
            "output": "Extracted 16 frames..."
          },
          {
            "step": 4,
            "role": "tool",
            "name": "object_detection",
            "output": "{\"detections\": [{\"object\": \"book\", \"confidence\": 0.95, \"bbox\": [100,200,300,400]}]}",
            "is_dinox": true,
            "detection_images": [
              "data/v12-detection-images/longvideobench_xxx_0/grid_step4_object_detection.jpg"
            ]
          }
        ]
      }
    }
  ]
}
```

**字段类型说明**：
- 整数字段：`total`, `correct`, `count`, `total_frames`, `input`, `output`, `image`, `thoughts`, `total`(token), `steps_count`, `step`
- 浮点字段：`rate`, `ratio`, `fps`, `duration`, `video_duration_seconds`, `duration_seconds`, `confidence`
- 布尔字段：`correct`(case), `is_dinox`
- 字符串字段：所有 ID、名称、文本内容

**图片路径**：JSON 中使用项目根目录相对路径（如 `data/v12-detection-images/qid/...`），HTML 渲染时前端代码添加 `../` 前缀构造实际 `<img src>`。

## 页面结构

### Section 00 — 概览统计（默认展开）

**准确率卡片**（水平排列）：
- Baseline 准确率：XX/121 (XX.X%)
- Tool 准确率：XX/121 (XX.X%)
- 改进/退步统计：+N↑ / -N↓

**Token 对比表**：
- 按 dataset 分组的 tool/direct 平均 token 及比率
- 列：Dataset | Tool 平均 | Direct 平均 | 比率

**视频时长分布**：
- 统计卡片：最短/最长/平均/中位数
- CSS 直方图：<1min / 1-3min / 3-10min / >10min 四个区间

### Section 01 — Case 分析（默认展开，核心）

**筛选栏**：
- 分类 Tab：`全部` | `改进↑ (tool_only)` | `退步↓ (direct_only)` | `都对 (both_correct)` | `都错 (both_wrong)`
- Dataset Tab：`全部` | `LongVideoBench` | `Video-MME` | `LVBench`
- 排序下拉：默认 | 按视频时长 | 按 token 差异

**Case 卡片**：

卡片头部（可点击折叠）：
- dataset badge（颜色区分三个 benchmark）
- 分类 badge（tool_only=绿色、direct_only=红色、both_correct=蓝色、both_wrong=灰色）
- question_id
- 视频时长 badge
- source version badge (v10/v12)

卡片摘要行（始终可见）：
- 问题文本（截断显示）
- GT | Baseline 答案 ✓/✗ | Tool 答案 ✓/✗
- Token: Baseline Xk → Tool Xk (×N.N)

折叠区 — Baseline 详情：
- Reasoning（marked.js 渲染 markdown）
- Response（marked.js 渲染 markdown）

折叠区 — Tool 执行轨迹：
- 按步骤渲染，每步包含：
  - AI 步骤：thinking 内容 + 工具调用列表
  - Tool 步骤：工具名称 badge + 输出内容
  - **DINO-X 步骤**（`is_dinox=true`）：
    - 特殊高亮边框（橙色左边线）
    - 相邻 AI 步骤的 thinking 内容突出显示
    - 检测可视化图片展示（`<img>` 引用，点击放大）
    - 检测结果文本

**加载状态**：页面加载时显示"加载中..."提示；fetch 失败时显示错误信息。

### Section 02 — 工具注册表 v12（默认折叠）

从 08 页面复制 `buildRegistry`、`filterRegistry`、`renderRegistryTools` 函数及对应 CSS，适配 v12 数据：
- 可折叠工具卡片
- 按 processing_level / backend 筛选
- DINO-X 相关工具（object_detection, instance_segmentation）高亮标注 v10→v12 变更

## 交互功能

1. **筛选 Tab 切换** — 按 category 和 dataset 过滤 case
2. **排序** — 默认 / 按视频时长 / 按 token 差异
3. **Case 折叠/展开** — 点击头部展开详情
4. **图片放大** — 模态框查看检测图片
5. **工具跳转** — 轨迹中工具名可点击跳转到 Section 02

## 技术实现

- 纯 HTML/CSS/JS，无构建工具
- CDN：marked.js（markdown 渲染）
- 图片：文件引用（`data/v12-detection-images/`），预估 ~15 个 case 有检测图片，每个 case 约 2-5 张图片，总计约 50-100 张 JPG
- 配色：沿用项目 5 大能力维度配色体系
- 图片不提交 git，通过 `.gitignore` 排除 `data/v12-detection-images/`，预处理脚本运行时生成

## 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/prepare_v12_analysis.py` | Python | 数据预处理脚本 |
| `data/v12-optimization-analysis.json` | JSON | 整合数据 |
| `data/v12-detection-images/` | 目录 | 检测可视化图片（gitignore） |
| `pages/09-v12-optimization-analysis.html` | HTML | 可视化页面 |
| `index.html` | HTML | 更新导航链接 |
| `CLAUDE.md` | Markdown | 更新项目结构和页面描述表 |
