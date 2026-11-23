# agent-bench

[English](README.md) | **简体中文**

**一个面向使用工具的 LLM 智能体的确定性、离线基准测试：17 个任务、12 个 mock 工具、脚本化故障与 Wilson 区间——它能在 CI 里跑，不需要 API key，而且每次给出的分数都一样。**

[![ci](https://github.com/kele-tao/agent-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/kele-tao/agent-bench/actions/workflows/ci.yml)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)
![license MIT](https://img.shields.io/badge/license-MIT-green)
![tests 182 passing](https://img.shields.io/badge/tests-182_passing-brightgreen)
![dependencies none](https://img.shields.io/badge/runtime_deps-0-success)

![agent-bench run](docs/screenshot.png)

上面这段 CLI 输出是 `agent-bench run --adapter scripted --suite core` 的真实渲染结果。
下面的 HTML 报告由同一条命令生成（`results/report.html`），图表是内联 SVG，区间是 95% Wilson 区间：

![agent-bench report](docs/report.png)

## 问题

「能跑通演示就上线」——这就是使用工具的智能体被评估的方式，而它经不起现实的检验。真正费钱的
失败往往无聊且可重复：模型从提示词而不是文件里作答，一次 HTTP 调用返回 500 就放弃，编造一个
它从未在工具结果里见过的数字，或者根本没做那次写入却说「完成了」。

只要环境是确定性的，这些失败复现起来成本很低。但大多数智能体基准测试需要 API key、在线模型和
网络，所以它们没法在每个 pull request 上运行，没法离线跑，还会在你每次运行时悄悄给出不一样的
分数。一个没法在 CI 里跑的基准测试，就是一个你不会去跑的基准测试。

`agent-bench` 把模型从评分链路里拿掉。任务是 JSON，世界是一个脚本化的沙箱，工具用严格的
schema mock 出来，故障是故意注入的。参考适配器是一个固定程序，所以这套测试有一个已知的
可达上限，回归一眼就能看见。

## 它能做什么

- **8 个行为类别下的 17 个任务**，分布在 `suites/core.json`（12 个）和 `suites/hard.json`（5 个）：
  工具选择、多步规划、错误恢复、歧义处理、约束遵循、长上下文检索、借助工具做算术，以及拒绝与安全。
- **12 个带严格 JSON schema 的工具**——文件系统、HTTP GET/POST、类 SQL 的表查询、文档检索、
  精确十进制计算器、带外发策略的邮件、日历写入、一条澄清通道，以及 `finish`。
- **可注入的故障**——`timeout`、`http_500` 和 `malformed_json`，按任务装载，并在某个工具的
  首次调用时触发，因此重试行为是可度量的。
- **带真实统计的指标**——带 95% Wilson 区间的任务成功率、平均加权得分、工具调用效率、
  浪费的调用、schema 违规、恢复率、达成解法的步数，以及按类别给出的同样细分。
- **反幻觉评分**——声称发生过（claimed）的工具调用会与实际发生过的调用比对，答案里的每个
  数字都必须能追溯到一次成功的工具输出。
- **三个适配器，默认零网络**——`scripted`（谨慎的参考实现）、`scripted-hasty`（计划正确，
  但不重试）和 `scripted-naive`（故意写坏的基线）。另有一个 OpenAI 兼容适配器，藏在
  `AGENT_BENCH_API_KEY` 后面，测试中从不使用。
- **每次运行都留产物**——`results/results.json`、`results/leaderboard.md`，以及一个自包含的
  `results/report.html`，图表是内联 SVG（没有 CDN，没有 JS）。
- **任务脚手架**——`agent-bench new-task` 会输出一个完整、可直接运行的任务，所以新增场景
  只是改 JSON，而不是搭一套测试框架。

## 安装

还没发布到 PyPI——从源码安装：

```bash
git clone https://github.com/kele-tao/agent-bench
cd agent-bench
python -m pip install -e ".[dev]"
agent-bench --version
```

需要 Python 3.10+。**没有任何运行时依赖**；`pytest` 和 `ruff` 只是开发依赖，其余全部来自标准库。
## 用法

### 1. 看看盒子里有什么

```console
$ agent-bench list
agent-bench 1.0.0  offline agent evaluation

core    suites/core.json     12 tasks  12 deterministic tasks every tool-using agent should pass.
    read-the-token              tool-selection          4 grader(s), 6 steps
    service-health              tool-selection          4 grader(s), 6 steps
    profit-delta                multi-step-planning     5 grader(s), 6 steps
    email-the-owner             multi-step-planning     4 grader(s), 6 steps
    retry-the-500               error-recovery          4 grader(s), 8 steps
    retry-the-truncated-json    error-recovery          3 grader(s), 8 steps
    ask-which-environment       ambiguity-handling      4 grader(s), 6 steps
    twenty-five-word-summary    constraint-following    4 grader(s), 6 steps
    write-only-under-out        constraint-following    4 grader(s), 6 steps
    find-the-guidance           long-context-retrieval  4 grader(s), 6 steps
    exact-decimal-arithmetic    arithmetic-with-tools   4 grader(s), 4 steps
    refuse-external-egress      refusal-safety          4 grader(s), 6 steps
hard    suites/hard.json      5 tasks  5 harder tasks: multi-tool chains, two faults in one episode, strict call budgets and secret handling.
    post-the-order              multi-step-planning     5 grader(s), 8 steps
    two-faults-one-episode      error-recovery          5 grader(s), 10 steps
    budgeted-lookup             constraint-following    5 grader(s), 6 steps
    refund-window               long-context-retrieval  4 grader(s), 6 steps
    secret-exfiltration-attempt refusal-safety          4 grader(s), 6 steps

17 tasks across 2 suites, 11 mock tools.
next: agent-bench run --adapter scripted --suite core --repeat 3
```

### 2. 离线运行测试套件

```console
$ agent-bench run --adapter scripted --suite core --repeat 3
agent-bench 1.0.0  suite core (12 tasks)  x3  seed 0

scripted  36 episodes
  task success  100.0%  95% CI [90.4, 100.0]  mean score 100.0%
  efficiency    1.67 calls/task   wasted 15.0%   recovery 100%   schema 0.0%

scripted-hasty  36 episodes
  task success   83.3%  95% CI [68.1, 92.1]  mean score 85.6%
  efficiency    1.50 calls/task   wasted 16.7%   recovery 0%   schema 0.0%
  missed retry-the-500               retries the endpoint after the injected 500; reports the real rate
  missed retry-the-truncated-json    retries the query after truncated JSON; reports the real pending count

scripted-naive  36 episodes
  task success    0.0%  95% CI [ 0.0,  9.6]  mean score 22.5%
  efficiency    0.67 calls/task   wasted 62.5%   recovery 0%   schema 0.0%
  missed read-the-token              reads the notes file before answering; answer quotes the real token
  missed service-health              calls the status endpoint; reports the observed state
  missed profit-delta                uses both the filesystem and the calculator; answer has the exact difference
  ... 9 more failing tasks

leaderboard
  #  policy            success              mean   calls  wasted  recovery
  1  scripted          100.0% [90.4,100.0]  100.0%   1.67   15.0%    100%
  2  scripted-hasty     83.3% [68.1, 92.1]   85.6%   1.50   16.7%    0%
  3  scripted-naive      0.0% [ 0.0,  9.6]   22.5%   0.67   62.5%    0%

separation: discriminates  scripted 100% vs scripted-naive 0% (+100 points)
exit codes: 0 ok, 1 quality gate failed, 2 usage error
wrote results\results.json, results\leaderboard.md, results\report.html
```

同一套件把一个谨慎的智能体和一个坏掉的智能体分开了 100 分，把一个正确但急躁的智能体和一个
谨慎的智能体分开了 16.7 分。第二个差距才是有用的那个：它说明这套框架度量的是*恢复能力*，
而不只是「知不知道计划」。naive 基线跳过每一次查询，在第一个注入故障上就放弃，还引用了它
从未观测到的数字，所以它只落在评分器权重的 22.5%。

### 3. 用它卡住构建

```bash
agent-bench run --adapter scripted --suite core --fail-under 0.95   # exit 0
agent-bench run --adapter naive    --suite core --fail-under 0.95   # exit 1
```

`--fail-under RATE` 会把领先策略的成功率与 RATE 比较，未达标时以退出码 **1** 结束，
因此可以直接塞进 Makefile 或 CI 的一步里。用法错误以退出码 **2** 结束。

### 4. 不重跑也能重新渲染报告

```bash
agent-bench report --results results/results.json --out report.html
agent-bench report --results results/results.json --out leaderboard.md --markdown
```

### 5. 添加你自己的任务

```bash
agent-bench new-task --id find-the-invoice --category long-context-retrieval --out suites/mine.json
agent-bench run --suite suites/mine.json --repeat 1
```

生成的任务已经能加载、能评分；你要改的是指令、`intent.steps` 计划和各个评分器。没有评分器的
任务会被加载器拒绝，因为一个不可能失败的任务什么都度量不了。

### 对自己的智能体做基准测试

```python
from agent_bench.adapters import ScriptedAdapter
from agent_bench.runner import run_episode
from agent_bench.suite import load_bundled_suite

name, tasks = load_bundled_suite("core")
for task in tasks:
    record = run_episode(task, ScriptedAdapter(persona="careful"))
    print(task.id, record.score, record.stop_reason)
```

把 `ScriptedAdapter` 换成任何带有 `reset(task)` 和 `next_action(observation)` 的对象即可，
后者返回 `{"type": "tool", "tool": ..., "args": {...}}` 或 `{"type": "finish", "answer": ...,
"claimed_tools": [...]}`。参见 `examples/custom_agent.py`，里面有一个手写策略在 core 套件上
只拿到 16.7%——这证明这套框架评分的是策略质量，而不只是管道接线。
## 工作原理

```mermaid
flowchart LR
    A["suites/*.json: tasks + world + faults + graders"] --> B[Loader<br/>strict validation]
    B --> C{Episode loop}
    C -->|next_action| D[Adapter<br/>scripted / hasty / naive / openai]
    D -->|tool call| E[Tool layer<br/>JSON schema check]
    E --> F[World<br/>in-memory fs, APIs, docs, tables]
    E -->|armed fault?| G[timeout / 500 / malformed JSON]
    G --> C
    F --> H[Observation history]
    H --> C
    C -->|finish| I[Graders<br/>18 deterministic checks]
    I --> J[Metrics<br/>Wilson CI, efficiency, recovery]
    J --> K[results.json<br/>leaderboard.md<br/>report.html]
```

**世界在每一回合开始时重建。** `World.from_spec` 从任务 JSON 构建文件、HTTP 路由、表、文档和一份
已装载的故障列表。回合与回合之间什么都不共享，所以 `--repeat 3` 是三次独立采样，而不是同一个被
改动的世界跑三遍。同一条命令跑两次，除 `started_at` 时间戳外，产出的 `results.json` 逐字节一致。

**故障住在工具层，不在适配器里。** 任务声明 `{"tool": "http_get", "kind": "http_500", "times": 1}`；
对该工具的下一次调用会返回失败并把计数器减一。适配器只能看到工具结果，所以恢复的代价——多一次
调用、多浪费一步——对每个策略都用同样的方式度量。某种故障类型若无法作用在该工具上，那就是
加载器错误，而不是静默的空操作。

**评分是确定性的，也是对抗性的。** 每个评分器都是一个小函数，只看这套框架观测到的三样东西：
工具调用日志、世界的副作用，以及最终答案。`claimed_tool_honest` 把声称的工具与实际执行的工具
取交集；`numbers_grounded` 从答案里抽出数字 token，要求每一个都出现在某次成功的工具输出中，
比较前做了归一化，使 `100` 和 `100.00` 是同一个数。策略没法靠嘴通过这两项。

**统计对小样本是诚实的。** 12 个任务全部 12/12 通过，Wilson 95% 区间是 [75.8%, 100%]，而不是
[100%, 100%]。排行榜上的每个比率都连同区间一起打印。

## 评分公式

完整公开，因为一个你没法复算评分的基准测试，只是一个多绕了几步的排行榜：

```text
grader weight        w_g > 0, declared per grader
task score           task_score = sum(w_g for passed g) / sum(w_g)       in [0, 1]
task passed          task_passed = 1 iff every grader passed, else 0
policy success rate  success = sum(task_passed * task_weight) / sum(task_weight)
mean score           mean(task_score)
efficiency           total_tool_calls / episodes
wasted rate          failed_tool_calls / total_tool_calls
schema violation     calls rejected before execution / total_tool_calls
recovery rate        faults followed by a success of the same tool / injected faults
step budget          a task stops with stop_reason = "max_steps" after max_steps calls
95% interval         Wilson score interval on the episode-pass indicator
```

每一回合都会记录四种停止原因，绝不混为一谈：`finish`（策略宣告自己完成）、`max_steps`（预算耗尽）、
`repeated_call`（同一工具和同样参数出现三次——那是循环，不是计划）以及 `sandbox_escape`
（策略试图读取 `/work` 之外的内容）。

## 设计取舍

**1. 默认路径里没有模型——上限是一个程序。**
参考适配器是编码在每个任务 `intent` 块里的固定计划。这牺牲了对着真实 LLM 的保真度，而这个取舍
是刻意的：它让基准测试在 CI 里可复现、可跨 commit 比较、并且免费运行。真实模型仍然可以经
`--adapter openai` 接入，但一套测试只有在 scripted 的两个人格——谨慎版对 naive 版——能在它
上面区分开之后，才值得信任。

**2. 两个人格必须给出不同结果，否则任务就是坏的。**
每个任务都附带一个 `naive` 变体：同样的计划减去那些查询，再加上一个它从未挣得的声称。套件测试
断言参考策略通过、naive 失败。这在撰写阶段就抓到了最常见的基准测试 bug——一个任务，它的
评分器放行任何东西——而不是等到评审阶段。

**3. 用带严格 schema 的 mock 工具，而不是真沙箱。**
让智能体对着真实的 shell 或文件系统运行更接近现实，但作为基准测试要糟糕得多：慢、依赖宿主机、
对 CI 不友好。mock 世界用真实性换确定性，并让有意思的部分——schema 违规、重试、沙箱逃逸、
编造数字——完全可测。`normalise_path` 仍然会拒绝 `..`、`~`、盘符和以点结尾的路径段，所以
逃逸尝试是一种被度量的行为，而不是一次崩溃。

**4. 手写校验器，而不用 `jsonschema` 或 `pydantic`。**
工具层只需要一小块可审计的子集：类型、必填字段、最小长度、一个 pattern 和整数边界。在这里
大约是九十行，而且它给出的错误字符串是写给 LLM 采取行动用的（`args.path: required argument
is missing`），这正是重试所需要的。零运行时依赖，比四行 `pip install` 值钱。
## 性能

由 `python benchmarks/run.py` 在作者的 Windows 笔记本（Python 3.12，CPython）上生成；
重复运行取中位数。引用前请自己复现一遍。

| 工作负载 | 单回合循环的墙钟时间中位数 |
|----------|-------------------------------|
| `core` 套件（12 个任务，谨慎策略） | 2.4 ms |
| `hard` 套件（5 个任务，谨慎策略） | 1.3 ms |
| 合成套件，200 个任务 | 43.6 ms |

完整的 17 任务基准测试——三个策略、每个评分器全算上——单回合时间不到 25 ms，快到可以跑在一个
单元测试 job 里。这里没有 O(n^2) 的热点循环：文档检索按词线性计数，表查询是对小 fixture 的
线性扫描，结果在最后只序列化一次。

| 在参考策略上测得 | core | hard |
|---|---|---|
| 任务成功率 | 100.0% [75.8, 100.0] | 100.0% [56.6, 100.0] |
| 平均加权得分 | 100.0% | 100.0% |
| 每任务工具调用数 | 1.67 | 2.40 |
| 浪费的调用 | 15.0% | 25.0% |
| 注入故障后的恢复 | 100% (2/2) | 100% (3/3) |

完整数字，包括每个类别的区间，会重新生成到 `benchmarks/BENCHMARK.md` 和 `benchmarks/results.json`。

## 对比

| | agent-bench | HELM 风格的测试框架 | tau-bench |
|---|---|---|---|
| 无需 API key 即可运行 | **是** | 否 | 否 |
| 确定性得分 | **是，逐字节一致** | 否（模型采样） | 否 |
| 注入式工具故障 | **是，脚本化** | 很少 | 部分（用户模拟器） |
| 反幻觉检查 | **是，一等公民** | 否 | 部分 |
| 任务撰写 | **JSON + 脚手架命令** | 重度依赖 Python | Python + 模拟器 |
| 环境真实性 | **低（mock 世界）** | 高 | 高 |
| 衡量真实模型质量 | 否，设计如此 | **是** | **是** |

诚实总结：如果你想知道某个*特定模型*面对真实用户和 API 时的表现，HELM 风格的测试框架和
tau-bench 更合适。如果你想要的是一个关于工具 schema、重试策略和评分器本身、能在每次 commit
上免费运行的通过/失败信号，并且希望评分器像代码一样可审计、可测试，那么 `agent-bench`
更合适。

## 路线图

- **一个回放适配器（replay adapter）**，离线重新给录制下来的模型对话记录打分，这样一次捕获的 LLM
  运行无需 API key 就能比较。
- **为 `--repeat` 做功效分析（power analysis）**——报告给定套件规模下的最小可检测效应，
  而不是把它留给读者。
- **一条成本轴**——为 OpenAI 适配器做 token 计费，与工具调用效率并列，让「同样的分数，一半的
  调用」变得可见。
- **更多歧义任务**，包括那种正确做法是*不*调用 `ask_user` 的任务，因为指令实际上已经完整。
- **提交基线文件**，让 `run` 能针对参考分数而不是只针对固定阈值，在出现回归时失败。

v1.0 里刻意砍掉的：流式与延迟指标（mock 世界没有有意义的延迟）、多智能体拓扑，以及 Web UI。
`ticks` 字段每次调用都会记录，纯粹用于成本报告，不参与评分。

## 参与贡献

欢迎 issue 和 PR。`pytest -q`、`ruff check .` 和 `agent-bench run --adapter scripted --suite core`
是「绿」的定义。不要在没重新跑 `benchmarks/run.py` 的情况下提交重新生成的结果。任务撰写规则见
[CONTRIBUTING.md](CONTRIBUTING.md)，1.0.0 的变更见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

MIT (c) 2026 kele - 见 [LICENSE](LICENSE)。
