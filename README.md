# LV-Viewer

长视频理解研究可视化工具集，用于展示和对比多模型在长视频理解任务上的能力分析、实验结果与执行过程。

## 页面一览

| 页面 | 说明 |
|------|------|
| `index.html` | 首页，导航至各可视化页面 |
| `ability-analysis.html` | 核心页面：能力总览、论文方法分析、Case 对比（含 Plan 可视化） |
| `execution-trace-v1.html` | 执行追踪可视化，逐步展示模型推理过程 |
| `tool-extraction-v1.html` | 工具提取分析，按能力维度分组展示 |
| `case-showcase-claude-v1.html` | Claude 模型 Case 展示 |
| `univa-discussion.html` | UniVA 论文深度讨论 |

## 五大能力维度

- 🟢 视觉感知 (Visual Perception)
- 🔵 时间动态理解 (Temporal Dynamics)
- 🔴 高层推理 (High-Level Reasoning)
- 🟠 空间与物体动态 (Spatial Object Dynamics)
- 🟣 视听联合理解 (AudioVisual Joint)

## 数据结构

```
data/
├── data.json                    # 主数据集：能力、论文、Case
├── experiment_index.json        # 实验索引（模型、结果路径、Prompt 模板）
├── results/                     # 各实验的结果 JSON
├── templates/                   # 各实验的 System/User Prompt
├── long_video_understanding_tools.json
├── execution-trace-v1.json
└── tool-extraction-v1.json
```

## 本地运行

本项目为纯静态站点，无需构建，使用任意 HTTP 服务器即可：

```bash
# Python 方式
python -m http.server 8080

# 访问
open http://localhost:8080
```

## 技术栈

- 原生 HTML / CSS / JavaScript（无框架）
- CDN 依赖：marked.js、renderjson、KaTeX、Prism.js
- 字体：Crimson Pro、Noto Sans SC、JetBrains Mono
