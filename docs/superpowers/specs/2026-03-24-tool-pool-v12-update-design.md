# Tool Pool v12 + 116 Questions Update

## Goal

Update the external `video_agent_tool_pool.html` page with latest tool pool (v12, 33 tools), 116 questions with Gemini trajectories, YouTube links, and sequential renumbering. Output as `pages/09-tool-pool-v12.html`.

## Data Sources

| Data | File | Key Fields |
|------|------|------------|
| Tool registry v12 | `video_cases_tools/data/outputs/s4_tool_registry_v12.json` | 33 tools with name, description, input, output, implementation, processing_level, mcp_server |
| 116 questions + abilities | `video_cases_tools/data/outputs/s2_case_abilities_116.json` | 22 abilities, 116 cases grouped by benchmark (LVBench:37, LongVideoBench:38, Video-MME:41) |
| Trajectories | `video_cases_tools/data/outputs/121case-gemini_3.1or3.0-tool_v10orv12.jsonl` | 121 records, filter to 116 matching questions. Fields: question_id (prefixed), trajectory, model, source_model, source_tool_version, steps, correct, agent_answer, ground_truth |
| YouTube URLs | `lv-viewer/data/execution-comparison-v2.json` | 121 youtube_url entries, all 116 questions covered (match by stripping dataset prefix from question_id) |
| Original page template | `/tmp/video_agent_tool_pool.html` | CSS + HTML structure + JS rendering functions |

## Data Transformations

### 1. REGISTRY object
- Read `s4_tool_registry_v12.json`
- Map each tool to original page format:
  - Keep: `name`, `description`, `input`, `output`, `processing_level`
  - Map `implementation`: keep `type` (single_model/pipeline/script), extract `backend` as model info, keep `notes`
  - Add `implementation.models` array from `mcp_server` + `backend` info
- Compute `usage` from trajectory data: per-tool frequency and ability_ids coverage
- No `provenance` needed

### 2. ABILITIES array
- From `s2_case_abilities_116.json`.abilities (22 items)
- Add `ability_category` mapping (same 5 categories as original)
- Add `ability_description` (brief, can derive from context)

### 3. TRAJ_DATA object
- Filter 121 JSONL records to 116 matching questions (by stripping dataset prefix)
- For each question:
  - `question_id`: original short ID from s2_case_abilities_116
  - `sequential_id`: 1-116 (new sequential number)
  - `ability_id`: from s2_case_abilities_116
  - `question_text`: from trajectory data
  - `benchmark`: from trajectory dataset field
  - `youtube_url`: from execution-comparison-v2.json
  - `trajectories`: single model entry with steps parsed from trajectory array
- Parse trajectory: extract tool_calls from assistant messages → generate steps [{step, tool, purpose, depends_on}]
- metadata: models list, total questions, successful trajectories

### 4. TRAJ_STATS object
- Computed from TRAJ_DATA:
  - `models`: per-model stats (avg_steps, total_steps, total_tools_used, trajectories count)
  - `per_tool`: per-tool per-model frequency
  - `per_tool_total`: per-tool total frequency

### 5. Question Numbering
- Sort: by benchmark (LVBench → LongVideoBench → Video-MME), then by ability_id, then by question_id
- Assign sequential_id 1-116

## Page Structure Changes

### Header
- Tag: "Tool Pool v12 + Trajectories"
- Title: "视频理解工具池 v12 & 轨迹概览"
- Subtitle: "33 Tools (13 Image · 14 Sequence · 6 Audio) × 22 Abilities × 116 Questions"

### Section 05 (轨迹详情) Layout Change
- Original: 3-column model comparison (GPT/Claude/Gemini)
- New: single-column trajectory display per question
- Show model info badge: `source_model` + `source_tool_version`
- Show correctness: correct/wrong badge with ground_truth vs agent_answer
- Add YouTube button in question header (red `youtube-btn` style from 08-execution-analysis-v2.html)

### CSS Additions
- `.youtube-btn` style (from 08-execution-analysis-v2.html)
- `.correct-badge` / `.wrong-badge` for answer correctness
- `.model-version-badge` for source_model + tool_version display
- Remove 3-column `.model-compare` grid, use single column

### Constants Update
- `MODEL_COLORS`: 2 Gemini variants instead of 3 different models
- `MODEL_SHORT`: descriptive labels for gemini-3.1-pro-preview / gemini-3-pro-preview
- Remove GPT/Claude color references

## Implementation Plan

### Step 1: Python data processing script
`scripts/build_tool_pool_v12.py` — reads all sources, produces 4 JSON data blobs

### Step 2: HTML template
Keep original page CSS/HTML structure, update:
- Header text
- Model constants
- Replace 3-col layout with single-col + correctness badges + YouTube buttons
- Embed generated data

### Step 3: Generate `pages/09-tool-pool-v12.html`
Script outputs final self-contained HTML

## Output
- `pages/09-tool-pool-v12.html` — self-contained HTML page
- `scripts/build_tool_pool_v12.py` — reproducible build script
- Update `index.html` with link to new page
