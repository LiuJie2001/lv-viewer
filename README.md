# LV-Viewer

长视频理解研究可视化平台，用于展示和对比多模型（GPT / Claude / Gemini）在长视频理解任务上的能力分析、实验结果与执行过程。

## 页面一览

| 编号 | 页面 | 说明 |
|------|------|------|
| — | `index.html` | 首页导航 |
| 01 | `pages/01-ability-analysis.html` | 能力总览 + 论文方法分析 + Case 对比（含 Plan 可视化） |
| 02 | `pages/02-univa-discussion.html` | UniVA 论文深度讨论 |
| 03 | `pages/03-case-showcase.html` | Claude 模型 Case 展示 |
| 04 | `pages/04-execution-trace.html` | 执行追踪可视化，逐步展示模型推理过程 |
| 05 | `pages/05-tool-extraction.html` | 工具提取分析，多模型对比 × 22 能力维度 ★ 主模板 |
| 06 | `pages/06-tool-pool.html` | 工具池 v7 + 轨迹概览（自包含，可离线查看） |

## 五大能力维度

- 🟢 视觉感知 (Visual Perception)
- 🔵 时间动态理解 (Temporal Dynamics)
- 🔴 高层推理 (High-Level Reasoning)
- 🟠 空间与物体动态 (Spatial Object Dynamics)
- 🟣 视听联合理解 (AudioVisual Joint)

## 项目结构

```
lv-viewer/
├── index.html                              # 首页导航
├── pages/                                  # 可视化页面（按创建时间编号）
│   ├── 01-ability-analysis.html
│   ├── 02-univa-discussion.html
│   ├── 03-case-showcase.html
│   ├── 04-execution-trace.html
│   └── 05-tool-extraction.html             ★ 主模板
│       └── 06-tool-pool.html                   ★ 自包含
├── data/                                   # 数据文件（统一 kebab-case）
│   ├── abilities-papers-cases.json         # 核心：能力/论文/Case
│   ├── tool-taxonomy.json                  # 21 工具分层体系
│   ├── experiment-index.json               # 实验索引
│   ├── execution-trace.json                # 页面 04 专用
│   ├── tool-extraction.json                # 页面 05 专用
│   ├── results/                            # 实验结果
│   │   ├── exp-001-baseline.json
│   │   └── univa.json
│   └── templates/                          # Prompt 模板
│       ├── exp-001-baseline/
│       └── univa/
└── images/
    ├── overview/                           # 论文概览图
    └── univa/                              # UniVA 论文图
```

## 工具提取模板使用指南（05-tool-extraction）

**核心可复用模板**：用于可视化多模型工具提取对比分析。

### 数据生成流程

1. **准备题目**：长视频理解题目，覆盖 22 项能力 × 5 大类
2. **提取阶段**：多模型分别对每题生成工具调用方案（prompts 定义见数据文件 `prompts` 字段）
3. **归并阶段**：用 reconcile prompt 将原始工具归并为注册工具
4. **输出**：生成符合格式的 JSON 数据文件

### 数据格式

```json
{
  "metadata": { "generated_date", "version", "models", "total_questions", "total_registry_tools" },
  "prompts": { "system", "user_template", "reconcile" },
  "abilities": [{ "ability_id", "ability_name", "ability_category", "ability_description" }],
  "questions": [{ "question_id", "ability_id", "question_text", "model_tools": { ... } }],
  "registry": [{ "name", "description", "input", "output" }]
}
```

### 创建新分析

1. 按上述格式生成 JSON 数据文件 → `data/<name>.json`
2. 复制 `pages/05-tool-extraction.html` → `pages/NN-<name>.html`
3. 修改 `fetch` 路径指向新数据文件
4. 在 `index.html` 添加链接

## 本地运行

纯静态站点，无需构建：

```bash
python /root/paddlejob/workspace/env_run/output/bwh/myhttp3.py <port>
# 访问 http://<bond0-ip>:<port>/index.html
```

## 命名规范

- **文件名**：kebab-case（如 `tool-extraction.json`）
- **页面编号**：`NN-` 前缀，按创建时间排序
- **数据文件**：描述性命名，版本由 git 管理

## 技术栈

- 原生 HTML / CSS / JavaScript（无框架）
- CDN 依赖：marked.js、renderjson、KaTeX、Prism.js
- 字体：Crimson Pro、Noto Sans SC、JetBrains Mono
