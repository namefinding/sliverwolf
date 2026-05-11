<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey.svg" alt="Windows">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <h3 align="center">🤖 Silverwolf — 本地 AI 智能管家</h3>
  <p align="center">一个跑在 Windows 上的本地 AI Agent，能听懂你说话、操作文件、上网搜索、收发 QQ 消息，还会记住你的偏好。</p>
</p>

---

## ✨ 它能做什么

- 📁 **管理文件** — "帮我把桌面上这周的周报整理一下，写个摘要" → 自动定位文件、读内容、写摘要
- 🌐 **上网搜索** — "查一下 OpenAI 今天发布了什么" → 打开浏览器搜索、抓取网页、整理结果
- 💬 **QQ 消息** — "看看 QQ 上谁找过我" → 查聊天记录、总结消息要点
- ⏰ **定时提醒** — "半小时后提醒我开会" → 到点推送
- 🎤 **语音交互** — 喊"嗨银狼"唤醒，直接说话下指令，TTS 语音回复
- 🧠 **记住偏好** — "以后搜索默认用 Bing" → 记住了，下次照办
- 📄 **文档处理** — 读写 Word/Excel/PPT，编辑、总结、格式转换

## 🧱 核心理念

一条清晰的边界贯穿整个项目：

```
LLM（大模型）   →  只管 理解、规划、说人话
工具模块       →  只管 校验参数、干活、返回结果
Kernel（内核） →  只管 调度、验证、存证
```

**LLM 会犯错，代码不会。** 所以每个决策都要过代码层的验证链——参数不对就拦截，路径不存在就驳回，任务没做完就不许说"做好了"。这套设计让本地小模型也能稳定运行。

## 🏗️ 架构

```
用户输入（文字 / QQ / 语音唤醒）
        │
   ┌────▼────────────────────────────┐
   │         Agent Kernel            │
   │                                 │
   │  意图分析 → 合同规划 → 执行循环   │
   │           │                     │
   │    ┌──────▼───────┐             │
   │    │   验证链       │             │
   │    │ Validator    │   ← 参数校验  │
   │    │ Critic       │   ← 决策审查  │
   │    │ ExecCritic   │   ← 完成检查  │
   │    └──────┬───────┘             │
   │           │                     │
   │    ┌──────▼───────┐             │
   │    │ 70+ 工具      │             │
   │    │ 文件·网页·QQ  │             │
   │    │ 文档·图片·系统 │             │
   │    └──────────────┘             │
   └─────────────────────────────────┘
        │
   ┌────▼────────────────────────────┐
   │  Memory 三层记忆  │  Trace 全链路 │
   │  SQLite 持久化    │  JSONL 审计   │
   └─────────────────────────────────┘
```

## 🚀 快速开始

### 前置条件

- **Python** >= 3.12
- **Ollama** — [下载安装](https://ollama.com)，然后拉一个模型：

```bash
ollama pull qwen2.5:1.5b
```

- **Windows** — 部分文件操作和桌面功能依赖 Windows 路径

### 安装

```powershell
# 克隆
git clone https://github.com/namefinding/sliverwolf.git
cd sliverwolf

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 安装
pip install -e .
```

### 配置

```powershell
Copy-Item config.example.yaml config.yaml
```

打开 `config.yaml`，改两个关键配置：

```yaml
agent:
  model: "qwen2.5:1.5b"           # 你 Ollama 里的模型名
  workspace_root: "C:/Users/你的用户名/Desktop/testing"  # Agent 的工作目录
```

### 启动

```powershell
# CLI 命令行模式
python -m local_agent.app.main --config config.yaml

# 或者双击
start_agent_server.bat
```

然后就可以打字跟它对话了。

### 可选的：启用语音

需要提前装好 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。

```yaml
voice:
  enabled: true                    # 开启 TTS 语音播报
  gptsovits_root: "C:/GPT-SoVITS" # 改成你的 GPT-SoVITS 路径

voice_input:
  enabled: true                    # 开启语音输入
  wake_word_enabled: true          # 开启"嗨银狼"唤醒词
```

### 可选的：接入 QQ

需要提前部署 [LLOneBot](https://github.com/LLOneBot/LLOneBot) 或其他 OneBot 实现。

```yaml
onebot:
  enabled: true
  ws_url: "ws://127.0.0.1:3001"   # OneBot 的 WebSocket 地址
```

然后启动 QQ Bot 网关：

```powershell
start_qq_bot_gateway.bat
```

## 📦 项目结构

```
src/local_agent/
├── kernel/              ← 核心引擎（编排循环、验证链、意图分析）
├── intent/              ← 意图分析服务
├── modules/             ← 70+ 工具（文件、网页、QQ、文档、图片、系统）
│   ├── file/            ← 文件读写搜索复制移动删除
│   ├── web/             ← 浏览器搜索与网页抓取
│   ├── qq/              ← QQ 消息记录收发
│   ├── retrieval/       ← 本地文件语义检索
│   ├── document_agent/  ← Word/Excel/PPT 读写编辑
│   ├── image/           ← 图片描述与 OCR
│   ├── system_utility/  ← 时间查询、定时提醒、定时任务
│   └── ...
├── memory/              ← 三层记忆（Hot/Warm/Cold）
├── storage/             ← SQLite 记忆 + JSONL Trace + 审计报告
├── skills/              ← 可插拔的 Skill 扩展包
├── voice/               ← Whisper ASR + GPT-SoVITS TTS + 唤醒词
├── protocol/            ← 数据模型与执行合同协议
├── llm/                 ← Ollama / DeepSeek 等 LLM 客户端
├── eval/                ← Trace 回放评估
├── runners/             ← Agent 入口调度
├── app/                 ← CLI / QQ Bot / 常驻服务器
└── workflows/           ← 预定义工作流
```

<details>
<summary>📊 项目规模</summary>

| 指标 | 数值 |
|------|------|
| Python 源文件 | 135 |
| 总代码行数 | ~44,000 |
| 测试用例 | 38 |
| 工具数量 | 70+ |
| 第三方 Agent 框架依赖 | 0 |

</details>

## 🔒 安全设计

| 机制 | 做了什么 |
|------|---------|
| 工作区沙箱 | 文件操作默认限制在 `workspace_root` 内 |
| 破坏性操作确认 | 删除文件、发 QQ 消息需要确认或代码拦截 |
| 参数硬校验 | 路径不存在 → 直接拒绝，不传给执行层 |
| LLM 幻觉拦截 | LLM 建议读一个不存在的文件 → Critic 驳回 |
| 过早完成拦截 | 要求产出文件但还没写入 → ExecutionCritic 不让结束 |
| 全链路 Trace | 每一步记录在 JSONL 里，出问题可以回溯 |

## 📋 待办 & 已知局限

- [ ] 多用户并发支持（目前是单用户单会话设计）
- [ ] 系统化的离线评估 Benchmark
- [ ] LLM 调用失败后的断点恢复
- [ ] DecisionCritic 工具跳转规则的自动生成
- [ ] 跨平台支持（目前部分功能 Windows only）

欢迎提 Issue 和 PR。

## 📄 License

MIT © 2025 Jin Zhenghao

---

<p align="center">
  <sub>Built with ❤️ in Shanghai · 全部手写，不调包搭 Agent</sub>
</p>
