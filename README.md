# Silverwolf — 本地 AI Agent

一个运行在 Windows 本地的 AI Agent 系统。基于 Ollama 部署本地大模型，支持文件操作、网页搜索、QQ 消息收发、语音交互等功能。

> 个人自用项目，仍在开发中。代码仅代表当前进展，不保证稳定可用。

## 功能

- 文件操作：读取、写入、搜索、复制、移动、删除、批量处理
- 网页搜索：浏览器搜索、网页抓取、内容提取
- QQ 集成：通过 OneBot 协议收发消息、查询聊天记录
- 文档处理：Word/Excel/PPT 的读取与编辑
- 图片处理：图片描述、OCR 文字识别
- 系统工具：时间查询、定时提醒、定时任务
- 语音交互：唤醒词检测、Whisper ASR 语音转文字、GPT-SoVITS TTS 语音播报
- 记忆系统：用户偏好与经验教训的持久化存储与自动召回

## 架构

```
用户输入（CLI / QQ / 语音）
        │
   ┌────▼────────────────────────────┐
   │         Agent Kernel            │
   │                                 │
   │  意图分析 → 合同规划 → 执行循环   │
   │           │                     │
   │    ┌──────▼───────┐             │
   │    │   验证链       │             │
   │    │ Validator    │  参数校验     │
   │    │ Critic       │  决策审查     │
   │    │ ExecCritic   │  完成检查     │
   │    └──────┬───────┘             │
   │           │                     │
   │    ┌──────▼───────┐             │
   │    │ 工具模块       │             │
   │    │ 文件·网页·QQ   │             │
   │    │ 文档·图片·系统  │             │
   │    └──────────────┘             │
   └─────────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │  Memory 三层记忆  │  Trace 全链路 │
   │  SQLite 持久化    │  JSONL 记录   │
   └─────────────────────────────────┘
```

**核心设计**：LLM 负责理解意图和规划，工具模块负责参数校验和确定性执行，Kernel 负责编排调度和状态管理。LLM 的每个决策都要过代码层的验证链——参数校验不通过则拦截，文件路径不存在则驳回，任务未完成则不允许声称完成。

**意图分析**：用户输入先经过正则快速匹配（识别本地路径、QQ 关键词、网页搜索等信号），再经 LLM 并行分析 6 个维度（知识类型、文档需求、站点搜索、任务拆解、记忆候选、指令识别），最终合并为一份 TaskEnvelope——规定本次任务的目标、可用工具族、必须产出、执行约束。这份合同在整个 turn 执行期间生效，后续每一步都受其约束。

**多层验证链**：
- `DecisionValidator`：对 70+ 工具各写了一个参数校验方法，同时检查路径安全性
- `DecisionCritic`：调用 LLM 二次审查 Planner 的决策，检查建议的文件路径是否在已知路径集合中，工具调用过渡是否合法
- `ExecutionCritic`：在任务声称完成前检查 required_outputs 是否真的全部产出
- `CompletionJudge`：对比 TaskEnvelope 的 required_outputs 与已完成的 outputs，判定任务是否结束
- `LoopController`：检测重复调用、连续失败，控制最大执行步数

**记忆系统**：分三层——
| 层级 | 存储 | 内容 | 召回方式 |
|------|------|------|------|
| Hot Context | 内存 | 当前对话摘要 | 直接拼入 prompt |
| Warm Memory | SQLite | 用户偏好、纠错、经验教训 | 按输入做 CJK bigram 分词检索 |
| Cold Archive | SQLite | 历史会话归档 | 仅在"之前""上次"等关键词触发时检索 |

Warm Memory 的存储类型包括 `user_fact`（用户事实）、`preference`（偏好）、`correction`（纠错）、`failure_pattern`（失败教训）、`success_pattern`（成功模式）等。超过 80 条学习记忆时自动压缩为 `lesson_digest`。

**Trace 系统**：每次 turn 以 JSONL 格式记录 20+ 种事件（user_input → intent_context → decision_raw → decision_review → tool_request → tool_result → completion_check → final_response）。每次执行后自动生成 Markdown 格式的 Trace Audit 报告，包括时间线、工作流节点状态、自动检测的 warnings。异常终止时 Self-Diagnosis 自动收集工具执行结果，构造诊断 prompt 分析根因。

**工具系统**：所有工具统一在 `ToolRegistry` 中注册，每个工具通过 `ToolManifest` 定义名称、描述、参数 schema、产出类型、安全标记（destructive / requires_confirmation）、权限模式（ALLOW / ASK / DENY）。破坏性操作需确认或直接拦截。Skill 系统支持将多步操作打包为单个工具，扫描 `skills/` 目录自动发现和注册。

**语音交互**：支持麦克风持续监听、唤醒词检测（"嗨银狼"）、Whisper ASR 语音转文字、VAD 语音活动检测、GPT-SoVITS TTS 语音合成。`LiveTurnState` 将多个连续语音片段合并为一次完整 turn 后再提交给 Kernel。

## 快速开始

### 环境要求

- Python >= 3.12
- [Ollama](https://ollama.com) 已安装并拉取模型
- Windows 系统

### 安装

```powershell
git clone https://github.com/namefinding/sliverwolf.git
cd sliverwolf
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### 配置

```powershell
Copy-Item config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
agent:
  model: "qwen2.5:1.5b"
  workspace_root: "C:/Users/你的用户名/Desktop/testing"
```

### 启动

```powershell
python -m local_agent.app.main --config config.yaml
```

### 可选：启用语音

需提前安装 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，然后在 `config.yaml` 中设置 `voice.enabled: true` 和 `voice_input.enabled: true`。

### 可选：接入 QQ

需提前部署 OneBot 实现（如 [LLOneBot](https://github.com/LLOneBot/LLOneBot)），然后在 `config.yaml` 中设置 `onebot.enabled: true` 和 WebSocket 地址，运行 `start_qq_bot_gateway.bat`。

## 项目结构

```
src/local_agent/
├── kernel/              # 核心引擎（编排循环、验证链、意图分析）
├── intent/              # 意图分析服务
├── modules/             # 工具模块
│   ├── file/            # 文件读写搜索复制移动删除
│   ├── web/             # 浏览器搜索与网页抓取
│   ├── qq/              # QQ 消息记录收发
│   ├── retrieval/       # 本地文件语义检索
│   ├── document_agent/  # Word/Excel/PPT 读写编辑
│   ├── image/           # 图片描述与 OCR
│   ├── system_utility/  # 时间查询、定时提醒、定时任务
│   └── memory/          # 记忆写入与召回
├── memory/              # 三层记忆（Hot/Warm/Cold）
├── storage/             # SQLite 存储 + JSONL Trace + 审计报告
├── skills/              # Skill 扩展包
├── voice/               # ASR + TTS + 唤醒词
├── protocol/            # 数据模型与执行合同协议
├── llm/                 # LLM 客户端（Ollama / DeepSeek）
├── eval/                # Trace 回放评估
├── runners/             # Agent 入口调度
├── app/                 # CLI / QQ Bot / 常驻服务器
└── workflows/           # 预定义工作流
```

## 已知局限

- 单用户单会话设计，不支持多用户并发
- 缺少系统化的离线评估 Benchmark
- LLM 调用失败后无断点恢复
- 部分功能仅支持 Windows
- 测试覆盖率不高

## License

MIT
