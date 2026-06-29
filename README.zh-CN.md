<p align="center">
  <img src="web/public/favicon.svg" alt="Bibilab" width="80" />
</p>

<h1 align="center">Bibilab</h1>

<p align="center">
  一个本地、私有的 <strong>视频版 NotebookLM</strong> —— 把视频与合集变成可搜索、
  带引用的 AI 笔记本。无需云端。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>中文</b>
</p>

把视频内容转成可搜索、AI 辅助的 **私人笔记本**。本地的 FastAPI 后端跑完整处理流水线
(download → transcribe → punctuate → chunk → digest ∥ embed),React + TypeScript
SPA 作为用户界面。单用户、单机、无云端。

> 本仓库包含三个包。各自的指南(下方链接)才是权威;本 README 是入口,以及查找
> 跨包配置的地方。
>
> | 路径 | 角色 | 自有指南 |
> |---|---|---|
> | `backend/` | FastAPI 流水线、SQLite、ChromaDB,生产环境由 FastAPI 托管 SPA | [`backend/CLAUDE.md`](backend/CLAUDE.md) |
> | `web/` | React + TypeScript SPA(Vite、Tailwind) | [`web/CLAUDE.md`](web/CLAUDE.md) |
> | `eval/` | RAG 答案评估框架(独立的 `uv` 包) | [`eval/CLAUDE.md`](eval/CLAUDE.md) / [`eval/README.md`](eval/README.md) |
> | `docs/` | 架构、引用系统、RAG 简介 | `docs/citation_system.md`、`docs/RAG简介.md`、`docs/roadmap.md` |

---

## Bibilab vs 谷歌 NotebookLM

同样是"与你的资料对话"的思路,但为那些想要 **本地、开源、视频原生** 而非云托管的人
而造。NotebookLM 是打磨精致的托管产品;Bibilab 用那份精致换取对自己数据和模型的
完全掌控。

|                                                      | Bibilab | NotebookLM |
| ---------------------------------------------------- | :-----: | :--------: |
| 完全本地运行,无需账号                                |    ✓    |     ✗      |
| 自托管 / OpenAI 兼容模型(Ollama、LM Studio)         |    ✓    |     ✗      |
| 开源                                                 |    ✓    |     ✗      |
| 支持 B 站及非 YouTube 视频入库                        |    ✓    |     ✗      |
| 带说话人标注的转写                                   |    ✓    |     —      |
| 回链到来源的行内引用                                 |    ✓    |     ✓      |

## 它能做什么

对每个入库的视频,Bibilab 会产出:

- 一份带标点、带说话人标注的转写
- **Sections(分节)** —— 有界的子来源跨度(按 token 量化,每节目标 ~12 000 tokens,
  范围 [7 200, 16 800]);短视频即一节。每节的摘要 + 关键词存在 `sections` 表;source
  行只携带 **facets**(系列、集数、季)。
- 每节的 chunks(中文目标 800 tokens,英文 300),适配向量 + BM25 混合检索。
- 一个按列表(per-list)的对话,带 **节级 `[N]` 引用**:LLM 在流式过程中调度
  `find_passages`(偏召回的定位器)或 `read_section(section_id="[N]")`(对单节的有界逐字
  读取),答案锚定在转写片段上,可点击跳回来源查看器中被引用的那一节。

存储按用户放在 `~/.bibilab/` 下(可用 `BIBILAB_HOME` 覆盖):SQLite(`bibilab.db`)存所有
结构化数据,ChromaDB(`chroma/`)存向量索引,`downloads/` 临时目录,`models/` 缓存本地
ASR / embedding / reranker 模型,`artifacts/{id}.md` 存生成的 artifact 内容。

---

## 环境要求

| 工具 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.12 | `backend` 和 `eval` |
| Node | ≥ 20 | `web` |
| [`uv`](https://docs.astral.sh/uv/) | 最新 | Python 包 + venv 管理器 |
| FFmpeg | 系统级 | 音频提取(`ffmpeg-python` 外部调用) |
| `aria2` | 系统级 | 多连接下载器(`apt install aria2` / `brew install aria2`);接入 yt-dlp 的 `external_downloader` 抑制单 IP 限速 |
| `yt-dlp` | 经 `uv` 自动安装 | B 站 / YouTube 适配器 |

> CUDA 可选。默认的 ASR / embedding / reranker 模型在 CPU 上运行;开启 GPU 加速只需
> 改一行配置(`transcription.device=cuda`)。

---

## 快速开始

```bash
# 1. 克隆
git clone <repo-url> bibilab && cd bibilab

# 2. 后端
cd backend
uv sync --dev --extra cpu         # 创建 .venv,安装全部依赖(cpu 版 torch)
# NVIDIA 机器改用 --extra cuda 以启用 GPU 加速转写。
uv run python -m bibilab.main     # 在 :8765 提供 API,生产环境同时托管 SPA
# 开发模式下单独跑 SPA —— 见第 3 步。
cd ..

# 3. Web(开发模式 —— 支持热更新 HMR)
cd web
npm install
npm run dev                       # Vite 跑在 :5173,把 /api 与 /proxy 代理到 :8765
```

开发模式打开 `http://localhost:5173`;若你先 `npm run build` 构建 SPA 并让后端托管,
则打开 `http://localhost:8765`。

首次运行会创建 `~/.bibilab/`,含一个空的 SQLite 库、一个配置文件,以及空的
`chroma/` / `models/` / `downloads/` 目录。本地模型(SenseVoice / Whisper、ONNX
MiniLM、BGE reranker)在首次使用时惰性下载 —— 第一次入库视频时预计需要 ~1–3 GB。

### Docker(一键部署)

在容器里构建并运行一切 —— 无需本地配置 Python/Node。

**前置条件:** Docker(含 Compose)。要启用 GPU 加速转写,需要 NVIDIA 驱动 +
支持 GPU 的 Docker —— **Docker Desktop(WSL2 后端)已内置**,而原生 Docker Engine
需要安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
没有 GPU 支持(或非 NVIDIA 主机)时,容器以 CPU 运行。

```bash
git clone <repo-url> bibilab && cd bibilab
./install.sh        # 一次性探测 GPU,构建匹配的镜像,启动容器
```

打开 `http://localhost:8765`。`install.sh` 探测 GPU 直通是否真正可用,自动选择
`cpu` 或 `cuda` 版 torch —— GPU 只加速 ASR 转写,所以 CPU 镜像功能完整,只是转写更慢。

| 主机 | 变体 |
|---|---|
| NVIDIA + 支持 GPU 的 Docker(Docker Desktop WSL2,或 Linux + Toolkit) | `cuda` —— GPU 转写 |
| 无 GPU、macOS、AMD/Intel GPU | `cpu` —— 全部走 CPU |
| Windows 原生(无 WSL) | 从 WSL/Git Bash 运行 `install.sh`;无 GPU 直通 |

即便误选了 `cuda` 也不会崩 —— 若 GPU 实际不可用,应用会在运行时回退到 CPU。
暂不支持 AMD/ROCm 加速。

数据存放在主机的 `~/.bibilab`(绑定挂载到 `/data`),**与原生安装共享** —— 同一套
DB、模型与配置。容器以你的主机用户身份运行,文件归属仍是你。更新时重新运行 `./install.sh`。

### Pre-commit 钩子

```bash
pip install pre-commit
pre-commit install                 # 一次性
```

对后端强制执行 `ruff check` / `ruff format`,并对全局强制去除行尾空白。

---

## 配置

配置位于 `~/.bibilab/config.json`(首次运行时创建)。完整 schema:

```jsonc
{
  "accounts": { "bilibili": { "cookie": "", "username": "", "avatar_url": "" } },
  "ai": {
    "protocol": "openai|anthropic",
    "model": "",
    "api_key": "",
    "base_url": null,
    "output_language": "ui",
    "context_window": 128000,
    "max_output_tokens": 16384
  },
  "transcription": {
    "model": "sensevoice-small|large-v3",
    "device": "cuda|cpu",
    "language": "auto"
  },
  "backend": { "port": 8765, "max_concurrent_jobs": 1, "cors_origins": ["*"] },
  "rag": {
    "max_distance": 0.8,
    "reranking_enabled": true,
    "hybrid_enabled": true,
    "debug_prompts": false
  }
}
```

- **`ai.api_key` + `ai.model`** —— 入库 / 对话所必需。后端通过 `AsyncOpenAI` /
  `AsyncAnthropic` 与任意 OpenAI 或 Anthropic 协议的端点通信。`base_url` 让你指向本地
  代理 / Ollama / vLLM。
- **`transcription.model`** —— `sensevoice-small`(中文,快)或 `large-v3`(Whisper,
  多语言)。模型在首次入库时下载。
- **`accounts.bilibili.cookie`** —— 部分 B 站视频需要(403 → 任务以 `needs_auth`
  结束)。使用 `/settings` 里的应用内扫码登录,或从浏览器粘贴 cookie 字符串。
- **`rag.debug_prompts`** —— 为 true 时,后端为每个对话轮次写一份 JSON 到
  `~/.bibilab/debug/{message_id}.json`;UI 会在这些助手气泡上显示 `</>` 图标,点击
  打开 prompt-trace 抽屉。默认关闭。

更适合命令行的设置方式是 FastAPI 端点;完整路由表见
[`backend/CLAUDE.md`](backend/CLAUDE.md)。

---

## 使用应用

1. **登录 / 连接账号** —— 打开 `/settings`。B 站点"扫码登录",用 App 扫码;cookie
   会保存到 `config.json`。
2. **创建列表** —— 列表是顶层的笔记本。在主页网格点"新建列表"。
3. **入库视频** —— 在列表详情页粘贴 B 站 / YouTube 链接。流水线在后台运行:
   `download → audio → transcribe → punctuate → derive_sections → chunk
    (per-section) → digest ∥ embed → write_source + write_transcript_segments
    + write_sections (atomic)`。在 **Jobs** 指示器上看进度。
4. **提问** —— 切到 **Chat** 标签。LLM 在流式过程中调度工具:
   - `find_passages(query, sequence_number?, season_number?)` —— 偏召回的定位器;
     返回 top-8 的 chunks,按 **section** 分组,置于
     `===== [N] "title" · Section M (mm:ss–mm:ss) =====` 围栏下,带说话人轮次重建。
     若传入 facet(如 `sequence_number=3`),则返回匹配来源的完整 **section 大纲** ——
     每节一个 `[N]`,只有摘要,在你下钻前不可引用。
   - `read_section(section_id="[N]")` —— 按引用序号对单节做有界逐字读取。当
     `find_passages` 显示某节切题、但片段没覆盖到所问的具体内容时使用。
   - facet 零匹配(或 facet-DB 出错)时,工具会 fail-open 到全量池,并标记
     `facet_scope.no_match=true`,让 LLM 在回答前说明范围已降级。
5. **下钻到来源** —— 来源详情查看器渲染 N 个 section 子块(每节一个)。对话里的引用
   chip 会打开来源并选中被引用那节的标签页。

RAG 双工具界面的细节(以及为何这个设计是无门控、双工具、节级的),见
[`docs/citation_system.md`](docs/citation_system.md) 与
[`docs/RAG简介.md`](docs/RAG简介.md)(中文,最初的设计文档)。

---

## 开发流程

### 跑三个测试套件

```bash
# 后端
cd backend
uv run pytest                       # 完整套件(含集成测试)
uv run pytest -m "not integration"  # 快速单测通道

# Web
cd ../web
npm test                            # vitest
npm run lint                        # tsc --noEmit

# Eval
cd ../eval
uv run pytest
```

### 目录结构(每个包一棵树)

- `backend/src/bibilab/` —— `routers/`、`pipeline/`(一阶段一文件)、`models/`、
  `adapters/`、`db.py`、`worker.py`、`config.py`、`cleanup.py`。完整目录树、数据库
  schema、流水线阶段、对话执行流程见 [`backend/CLAUDE.md`](backend/CLAUDE.md)。
- `web/src/` —— `components/`(ui 原语、auth、debug、lists、lab、hooks、jobs、
  layout、settings)、`pages/`、`lib/`(api 客户端、types、constants、i18n、chat
  工具)、`app/`、`test/`。见 [`web/CLAUDE.md`](web/CLAUDE.md)。
- `eval/src/eval/` —— `cli.py`(入口)、`models.py`、`config.py`、`generate.py`、
  `tui.py`、`runner.py`、`grader.py`、`reporter.py`、`dashboard.py`、`storage.py`。
  见 [`eval/CLAUDE.md`](eval/CLAUDE.md)。

### 提交 & PR 约定

提交信息:`"<type> | <scope> | #<issue> <description>"`。type 为
`feat | fix | refactor | chore | docs`。PR 标题同格式(适用时以 `#<issue>` 开头)。
先开分支 —— 永不直接提交到 `master`。

往一个未推送的提交上 amend 修复时,用
`git push --force-with-lease=<branch>:<expected-sha>`(钉死 SHA)而非裸 `--force`,
以免覆盖你 fetch 与 push 之间被推上来的工作。

### 测试中的 home 隔离

每个后端测试通过 `tmp_bibilab_home` fixture 设置 `BIBILAB_HOME` —— 一个
`bibilab_home()` 认得的环境变量接缝,让测试拿到每次运行独立的临时目录,无需逐模块
打补丁。SQLite、Chroma 和模型缓存都遵循它。

### LLM mock

触及 LLM 接缝的测试应使用 conftest fixtures `mock_stream_llm`(chat)和
`mock_call_llm`(digest / chat_summary / worker)。用 `.return_value` /
`.side_effect` 配置;不要逐测试手写 `patch("...stream_llm")`。命中真实 SQLite /
Chroma 的集成测试以模块级 `pytestmark = pytest.mark.integration` 标记,并被
`-m "not integration"` 排除。

---

## 项目特有约定(别破坏这些)

这些是新贡献者最容易无意违反的承重决策。理由在 [`CLAUDE.md`](CLAUDE.md);这里的
TL;DR 是给快速浏览者看的。

- **FastAPI 侧没有 `/api` 前缀。** 路由挂在根上;`web/vite.config.ts` 在开发模式
  代理到 `:8765` 时把 `/api/*` → `/*`。
- **没有来源级的 `summary` / `keywords` 列。** 它们已被移除(#465 之后)。唯一的
  digest 存储是 `sections` 表。
- **`db.py` 只放 SQL。** 不做状态映射,不放业务逻辑。领域推导放在 routers / service
  函数里(状态映射已抽到 `video_status.py`)。
- **Section ID 在 LLM 面向的界面是字符串,内部是 int。** 后端把整型的
  `sections.id` 序列化为 JSON number;SSE 入库路径强转为字符串,以便严格相等匹配
  (引用跳转、历史重载)生效。在审计每个入库边界之前,别重构掉这层强转。
- **每节一个 `[N]`**,不是每来源一个。`CitationRegistryEntry` 以 `section_id` 为键。
  仅大纲的 section 在 `read_section` 下钻它们之前是 `citable=False`。
- **跨节顺序按 rerank 首次命中;节内片段按时间顺序渲染(segment 顺序,非 rerank
  顺序),非 seg 相邻的片段之间用 `[…]` 间隔标记。** rerank 是跨节的排序权威,但围栏
  内 LLM 按口语顺序读该节自己的摘录。
- **代码里没有 migration。** schema 变更是一次性的 `sqlite3` CLI 调用;别为加
  migration 去改 `bootstrap_db`。
- **代码注释里没有外部引用。** 不写 `(#441)`、不写 `docs/specs/...` 路径、不写事件
  日期、不写 git log 引用。允许:仓库内交叉引用、`# noqa:`、纯文字的设计理由。memory
  文件是唯一可以引用 git 提交 / PR-# 的地方。
- **一次性脚本(migration、backfill、fixup)在验证运行后删除。** 它们是未跟踪的
  工具,不是接缝。

---

## 参与贡献

1. **挑一个 issue** —— 见 GitHub issues 列表。开放问题的路线图在
   [`docs/roadmap.md`](docs/roadmap.md)。
2. **从 `master` 开分支** —— `git checkout -b <type>/<scope>-<short-desc>`。
3. **动代码前先读相关的 `CLAUDE.md`** —— `Code Health Rules`(无死代码、无魔法字符串、
   无重复逻辑、数据层不放业务逻辑、范围纪律、提交前验证)适用于每一次改动。
4. **每个实现阶段后跑 `/simplify`。**
5. **提交前验证** —— `uv run pytest`(后端)与 `npm test && npm run lint`(web)必须
   通过。
6. **开 PR。** 在标题和正文里引用 issue 编号。落地时 squash-merge;issue 会自动关闭。

README 不重复的更深文档,见:

- [`docs/citation_system.md`](docs/citation_system.md) —— 完整的节级引用系统
- [`docs/RAG简介.md`](docs/RAG简介.md) —— 中文设计文档(入库 + 查询阶段、双工具界面、
  FTS vs 向量的取舍理由)
- [`docs/roadmap.md`](docs/roadmap.md) —— 按版本列出的开放问题
- [`backend/CLAUDE.md`](backend/CLAUDE.md) —— 后端约定、schema、流水线、对话执行
- [`web/CLAUDE.md`](web/CLAUDE.md) —— 前端约定、RAG 元数据形状
- [`eval/CLAUDE.md`](eval/CLAUDE.md) / [`eval/README.md`](eval/README.md) —— 评估框架

关于本 README 刻意省略的 agent 专属上下文(memory、feedback),见
`~/.claude/projects/.../memory/MEMORY.md` 和项目根的 [`CLAUDE.md`](CLAUDE.md)。

## 许可证

[MIT](LICENSE) © AmosChenZixuan
