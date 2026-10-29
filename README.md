# 银行制度智能问答（Policy RAG）

面向银行制度文档的检索增强问答系统。回答只使用**现行有效**的制度条款，
每条结论标注《制度全称》（发文字号）第 X 条，可展开核对原文。

## 解决什么问题

制度问答和通用文档问答的难点不在"能不能生成"，而在两件事：

1. **制度会换版。** 同一个问题，2021 年的旧办法和 2024 年的新办法给的数字不一样。
   如果检索层把新旧条款一起召回，模型会拿旧标准回答今天的问题，而且答案看起来毫无破绽。
2. **答案必须是可追溯的口径。** 一线柜面人员按错误额度放款、按废止条款做尽调，
   损失是实打实的。答案对错不由语气决定，必须落到具体条款上。

系统针对这两点做了两个硬约束：

- **时效硬过滤**：每份制度解析出发文字号、施行日期和废止关系。检索时按查询基准日
  过滤，已废止制度的条款根本不进候选池，不做"新旧并陈让模型自己选"。
- **引用校验**：生成后逐个校验答案中的数字是否出现在所引条款原文里，
  缺失即判为幻觉，触发一次重生成；仍不过关则降级为条款原文直读。

## 目录结构

```
bank_policy_rag/
├── core/
│   ├── config.py              配置读取（环境变量 > config.ini > 默认值）
│   ├── logger.py              统一日志
│   ├── qa_system.py           问答主链路：改写 → 检索 → 生成 → 校验
│   ├── ingest/
│   │   ├── parser.py          制度解析：文号、章节、条款、施行日期、废止关系
│   │   ├── chunker.py         父块=一条，子块=条内切片
│   │   └── embedder.py        BGE-M3 双向量（稠密 + 稀疏），含离线兜底
│   ├── rag/                   ★ RAG 问答层：问题到答案经过的全部环节
│   │   ├── faq/preprocess.py  FAQ 分词（jieba，含业务词表与停用词）
│   │   ├── faq_search.py      标准问法 BM25 匹配 + 三重闸门 + 时效复核
│   │   ├── query_analyzer.py  查询策略：直接 / 子查询 / 回溯 / 情境补全 / 历史版本
│   │   ├── retriever.py       混合检索 → 时效校验 → 去重 → 阈值截断
│   │   ├── reranker.py        交叉编码器精排（bge-reranker-large）
│   │   ├── prompts.py         系统提示词与上下文拼装
│   │   ├── llm_client.py      模型客户端（OpenAI 兼容接口）
│   │   └── answer.py          生成 + 引用校验 + 降级
│   ├── store/
│   │   ├── policy_db.py       制度、条款、标准问答三张表
│   │   ├── vector_store.py    Milvus 集合：稠密 + 稀疏双索引
│   │   ├── redis_client.py    Redis 通用客户端，三层缓存共用
│   │   └── qa_cache.py        第一层缓存：问答结果 answer:{问题}|{条线}|{基准日}
│   └── assessment/            评估三层，详见「效果评估」
│       ├── eval_set.py        32 条评估集，从制度原文反推
│       ├── runner.py          主环境跑全量题，产出回答与召回明细
│       ├── rag_as.py          RAGAS 四指标，含引用偏差对照
│       ├── ab_citation.py     引用格式对 faithfulness 影响的对照实验
│       └── diagnose.py        按题打印单项得分，定位掉分条目
├── scripts/
│   ├── ingest.py              入库脚本（支持 --check 预检 / --drop / --keep-vectors）
│   └── seed_faq.py            从条款自动生成 FAQ 候选，供人工审定
├── data/raw/                  制度原文（md / txt / pdf / docx）
├── data/faq/                  faq_seed.csv，审定后的标准问答
├── data/eval/                 评估产物：raw_results / ragas_report / per_item
├── tests/offline_check.py     离线自测：不依赖 MySQL / Milvus / 模型
├── web/static/index.html      前端页面
├── Dockerfile                 应用镜像（CPU 版 torch，2.1GB）
├── docker-compose.yml         应用 + MySQL + Redis + etcd + MinIO + Milvus
├── .env.example               环境变量模板
├── main.py                    命令行入口
└── api.py                     HTTP 接口
```

## 数据说明

`data/raw/` 下的四份制度为**示例数据**，用于演示解析规则与检索效果，包括：
文号带年份与序号、正文声明废止旧办法、章条两级结构、条款内枚举项等。

### 支持的文件格式

| 格式 | 读取方式 | 说明 |
| --- | --- | --- |
| `.md` / `.txt` | 直接读取 | 保留原有空行与段落，作为解析基准 |
| `.pdf` | PyMuPDF 读取文本层 | 按文本块取内容，丢弃页眉页脚；不依赖 OCR |
| `.docx` | python-docx | 读段落与表格，保留段落边界 |

三种格式走同一条解析链路：**取文本 → 还原折行与段落 → 抽元数据 → 切条款 → 切块**，
最终产出完全一致。同一份制度分别存成三种格式入库，得到的条款编号、正文与
元数据一字不差（可用 `tests/offline_check.py` 的解析校验验证这一致性）。

PDF 与 Word 的文本不是「干净」的，还原折行这一段专门处理三类排版损耗：

- **页眉页脚**：位于页面上下边缘、长度不超过 40 字的文本块，判定为页眉页脚丢弃，
  避免「恒信银行个人贷款管理办法」这样的页眉被当成制度标题
- **跨页断条**：一条制度可能断在两页之间（分项在第 1 页末、后续规定在第 2 页首），
  因此折行合并与段落补空行都在**全文层面**执行，不按页各自处理
- **粘连的机构全称**：PDF 里「恒信银行股份有限公司恒信银发〔2024〕27号」常连成一行，
  取文号时会把机构通用词（银行/公司/股份有限公司）从代字前面剥掉，只留「恒信银发」

若 PDF 是扫描件（无文本层），解析会明确报错并提示先做 OCR 转文本，
本项目不内置 OCR。

替换成真实制度文本时，只要保持 `文本 > ## 第X章 > ### 第X条 小标题` 的结构，
解析器即可直接复用；若原件本身就是 PDF 或 Word，放进 `data/raw/` 即可，无需先转格式。

## 快速开始

### 方式一：Docker（推荐，一条命令起全套）

MySQL、Redis、Milvus 三件套连同应用一起由 `docker-compose.yml` 拉起，不需要本机预装任何组件。

```bash
# 1. 配置密钥（.env 已在 .gitignore 中，不会被提交）
cp .env.example .env
#    编辑 .env，填 POLICY_LLM_API_KEY 与数据库/缓存密码

# 2. 构建并启动（首次构建约 3 分钟，之后改代码重建在 10 秒内）
docker compose up -d

# 3. 等全部容器 healthy
docker compose ps

# 4. 初始化 Milvus 库（仅在全新数据卷时执行一次）
docker compose exec -T app python -c "from pymilvus import MilvusClient; \
  c=MilvusClient(uri='http://milvus:19530'); \
  [c.create_database(d) for d in ['bank_policy'] if d not in c.list_databases()]"

# 5. 入库制度 + 灌 FAQ
docker compose exec -T app python -m scripts.ingest
docker compose exec -T app python -m scripts.seed_faq --apply

# 6. 打开 http://127.0.0.1:8000
```

模型权重不打进镜像，而是通过 `./models:/app/models:ro` 只读挂载。
首次运行前需把 `bge-m3` 与 `bge-reranker-large` 放到项目的 `models/` 下。
镜像里 torch 从 CPU 专用源安装（`--index-url .../whl/cpu`），
否则会连带拉入约 5GB 的 CUDA 依赖，镜像从 2.1GB 膨胀到 6.9GB。

常用命令：

```bash
docker compose logs -f app    # 跟日志
docker compose down           # 停止，数据卷保留
docker compose down -v        # 停止并清空数据（下次需重新 ingest）
```

### 方式二：本机直接跑

```bash
pip install -r requirements.txt

# 1. 入库（解析 → 切块 → 写 MySQL + Milvus）
python -m scripts.ingest

# 2. 命令行试问
python main.py -q "个人消费贷款单户额度上限是多少" --show-clauses

# 3. 起服务
python api.py   # http://127.0.0.1:8000
```

入库前可先跑预检，只报告解析结果不写库，避免"导入了 0 条却显示成功"：

```bash
python -m scripts.ingest --check
```

依赖组件：MySQL（存条款结构与效力状态）、Milvus（存双向量）、Redis（问答缓存）。
连接信息在 `config.ini` 中配置，密钥通过环境变量注入：

```bash
POLICY_LLM_API_KEY=sk-xxx        # 模型密钥
POLICY_MYSQL_PASSWORD=xxx        # 数据库密码
POLICY_REDIS_PASSWORD=xxx        # 缓存密码
```

密钥优先级为 环境变量 > config.ini > 默认值，**不要把密钥写进 config.ini**。


Redis 默认使用 `db = 3`，与 Milvus 元数据所在 db0 隔离；若与 Milvus 共用同一个
Redis 实例，务必保留这个隔离，否则两边键空间会互相干扰。

## 检索与生成链路

```
问题 → 缓存命中？（同条线、同基准日直接返回）
     → 查询策略判定（直接/子查询/回溯/情境补全）
     → FAQ 标准问法匹配（命中且复核未过期 → 只做条款回表校验，跳过召回）
     → BGE-M3 混合检索（稠密+稀疏，取 top_k=8）
     → 时效硬过滤（废止条款、尚未生效条款在此拦截）
     → 同条款去重 → 相似度阈值截断
     → bge-reranker-large 交叉编码器重排（取 top 3）
     → 生成 + 逐句引用校验 → 不过关则重生成 → 仍不过关则拒答
```

FAQ 在链路里排在检索之前，因为它的定位是"已知问题的快路径"，不是"更好的检索"。
但它绕过检索、也跳过了时效硬过滤，所以必须自己做条款回表校验——
命中却校验不过时**退回完整链路**，而不是硬返回。

重排这一步不是可选项。向量分衡量的是"问题和条款整体像不像"，交叉编码器衡量的是
"这条条款能不能回答这个问题"。实测中问"借钱最多能批多少"，向量分把"审批权限"
排在"贷款额度上限"前面，重排后后者以 0.910 对 0.729 反超，答案才落在正确的条款上。

## 当前状态与已知边界

这几条是刻意写明的，避免误用：

- **默认走离线哈希向量。** `config.ini` 里 `allow_hash_fallback = true` 时，缺少 BGE-M3
  权重会退化为字二元组哈希向量——管道能跑通，但**语义检索失效**，只能命中字面重合。
  生产使用必须下载 BGE-M3 到 `models/bge-m3/`，并把该开关改为 `false`。
  改开关后必须 `--drop` 重新入库，哈希向量与 BGE 向量维度不一致。
- **模型未配置时自动降级为条款原文直读。** 回答开头会注明"未取得模型生成结果"，
  此时引用校验不参与（走的是降级路径），问题定位能力低于完整链路。
- **Redis 未配置只是少一层缓存**，读写失败均 try/except 降级，不影响问答。
- **CPU 环境下首次问答较慢。** BGE-M3 与重排模型合计约 4.3GB，首次加载需十余秒；
  单次命令行调用会重新加载，起 `api.py` 长驻服务后仅首次加载，之后为毫秒级。
  问答结果有 Redis 缓存，同一问题（同条线、同基准日）二次提问直接命中。

不启动任何外部组件也能验证核心逻辑：

```bash
python -m tests.offline_check
```

它会解析全部制度、检查切块归属、用内存实现跑一遍检索以验证废止条款被拦住，
并验证引用校验能抓出编造的数字。

## 效果评估

评估分两层：**自研指标**管"检索对不对、该拒有没有拒"，**RAGAS** 管"回答有没有依据"。
两层都要跑，因为自研指标抓的是业务正确性，RAGAS 抓的是生成忠实度，互相替代不了。

### 评估集

`core/assessment/eval_set.py`，32 条，六类：

| 类型 | 条数 | 考察点 |
| --- | --- | --- |
| numeric | 11 | 金额、比例、天数能否原样引用（50万/500万/1000万/70%/55%/3个工作日） |
| factual | 5 | 定义与范围（受益所有人条件、禁止用途、风险等级分级） |
| procedural | 8 | 操作步骤与审批层级（开户核验、审批权限分档、催收留痕、挂失补办） |
| temporal | 4 | 新旧制度交替（旧暂行办法是否废止、当前执行哪个口径） |
| multi_target | 2 | 一个提问涉及多个条款（额度与期限同时问、跨文档） |
| refuse | 4 | 制度外问题是否稳定拒答（离职申请书、绩效工资涨幅、竞品利率） |

评估集是**从制度原文反推**的，不是凭印象写题再去找条款。
制作办法是先打印全部 54 条条款，逐条读完之后再出题，出完再用脚本校验每题指向的条款号是否与预期一致。
如果题目不是从制度原文来的，`context_recall` 衡量的就是题库质量，不是系统能力。

### 两层评估怎么跑

主环境（跑系统）与 `.venv-ragas`（跑 RAGAS）分开，因为 ragas 0.2 依赖
`langchain-core 0.3.x`，而主链路已升到 `1.7.0`，两者不能共存。

```bash
# 第一层：主环境跑完 32 题，产出回答与召回明细
python -m core.assessment.runner --no-cache
# → data/eval/raw_results.json

# 第二层：独立 venv 算 RAGAS 四指标
.venv-ragas/Scripts/python.exe -m core.assessment.rag_as
# → data/eval/ragas_report.json
```

`.venv-ragas` 的建立方式见文末「环境复现」。
评估用的 embedding 复用检索侧的 BGE-M3（`models/bge-m3/`），
不另建一套——否则"相关性"衡量的不是同一个语义空间。

### 第一层：检索与决策（自研指标）

`core/assessment/runner.py` 在跑完每道题后自查四件事：

| 指标 | 含义 | 实测 |
| --- | --- | --- |
| 条款召回命中率 | 期望条款是否出现在召回结果里 | **28/28 = 100%** |
| 拒答准确率 | 制度外问题是否拒答 | **4/4 = 100%** |
| 可答题作答率 | 可答题是否给出了回答而非误拒 | **28/28 = 100%** |
| 带条款引用 | 回答是否标注了出处 | **28/28 = 100%** |

这四项全部是**确定性判断**：期望条款号与召回列表做集合比对，
`refused` 字段取系统返回值。不依赖 LLM 评判，因此可重复、可回归。

### 第二层：RAGAS 四指标

`data/eval/ragas_report.json`，28 条可答题（拒答题不参与 RAGAS，理由见下）：

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| context_recall | **0.9643** | 该召回的条款全部召回 |
| context_precision | **0.9405** | 召回条款中绝大多数是相关的 |
| answer_relevancy | **0.8806** | 回答切题 |
| faithfulness（去引用口径） | **0.9452** | 回答的论断几乎全部有条款依据 |
| faithfulness（原始口径） | 0.7204 | 含引用格式带来的评估偏差 |
| citation_bias | **-0.2248** | 上述两者的差 |

### 为什么 faithfulness 要报两个数

制度问答要求每条结论句末挂《制度全称》（发文字号）第X条，这是审计要求，
不能为了指标好看而删掉。但这串编号**不在条款正文里**，RAGAS 逐句判断
论断是否有依据时，会把括号里的编号当成一句独立断言，找不到出处就判为无据。

做了对照实验（`core/assessment/ab_citation.py`）：同一批回答跑两遍，
一遍带引用括号、一遍去掉，只改评估输入不改系统输出。

```
带引用括号   faithfulness = 0.7204
去引用括号   faithfulness = 0.9452
差值         +0.2248
```

差值是引用格式对指标的系统性污染，不是回答里的幻觉。
因此报告同时给出两个口径，把偏差显式量化，而不是挑好看的报。

> 如果被问到"为什么不直接报 0.95"：因为删掉引用不符合制度问答的合规要求，
> 而指标低的原因是评估工具不认识引用格式。正确做法是报告口径差异，
> 让读的人知道 0.72 和 0.95 之间差的是什么，而不是二选一。

### 拒答题为什么不跑 RAGAS

拒答题的答案是"现行有效的制度中未检索到与您问题直接对应的条款"，
对照的 `ground_truth` 也是同一句话。RAGAS 的四个指标都在衡量
"回答与参考答案的语义贴合度"，两个拒答句子的贴合度天然接近天花板，
而 `contexts` 里塞的是不相关条款，指标必然接近 0——**这个 0 不代表系统坏了**。

拒答题要衡量的是"该拒的时候有没有拒"，这是分类问题不是生成质量问题，
所以单独统计 `refusal_accuracy`（当前 4/4 = 1.0），不混进 RAGAS 均值。

### 评估指标不能只看均值

`core/assessment/diagnose.py` 按题打印单项得分，用于定位是哪一条在掉分：

```bash
.venv-ragas/Scripts/python.exe core/assessment/diagnose.py
# → data/eval/per_item.csv
```

主报告只给均值，看不出分布。这套系统第一次跑出的 faithfulness 是 0.51，
按题打印后才发现是回答被 prompt 要求成"结论 + 要点 + 条款清单"三段式重复，
同一件事说了三遍，论断数膨胀、无据论断随之变多。改掉重复结构后升到 0.69（原始口径）。

## LangChain 在哪些环节、为什么

系统的设计取舍不是"用上框架"，而是**把可以替换的边界交给框架，把业务判断留在自己手里**。
下表按当前代码逐个文件列出实际用的组件：

| 环节 | 文件 | LangChain 组件 | 为什么在这用 |
| --- | --- | --- | --- |
| 文档解析 | `core/ingest/parser.py` | `BaseLoader` + `Document` | 每种文件格式一个 loader，`lazy_load()` 是统一契约；新增格式只加类，不改主链路 |
| 文本切分 | `core/ingest/chunker.py` | `RecursiveCharacterTextSplitter` | 按分隔符优先级递归切；中文标点需重写分隔符表（见下） |
| 提示词 | `core/rag/prompts.py` | `PromptTemplate` | 变量声明成契约，漏传直接抛错，而不是 f-string 拼出残缺 prompt 静默跑下去 |
| 模型调用 | `core/rag/llm_client.py` | `ChatOpenAI` | OpenAI 兼容协议，换端点只改 `base_url`；`chain()` 输出可接进 LCEL 管道 |
| 查询理解 | `core/rag/query_analyzer.py` | `PromptTemplate` + `StrOutputParser` | 四个改写模板走「模板 \| 模型 \| 解析器」管道，换模型不动模板 |
| 向量库 | `core/store/vector_store.py` | `VectorStore` 子类 | `as_retriever()` 让自研混合检索进 LangChain 生态，同时保留自己的过滤逻辑 |
| 缓存 | `core/store/qa_cache.py` | `BaseCache` + `Generation` | 模型调用可走缓存；业务语义键与 LangChain 键空间分开 |

### 三处刻意不交给框架的地方

**一、多路召回融合。** LangChain 的 `Milvus` 向量库实现走单路 `similarity_search`。
本项目需要稠密向量 + 稀疏向量两路同时召回，再按维度归一化后加权融合——融合权重
是要调的业务参数，包进框架反而看不见它在做什么。所以只继承 `VectorStore` 实现接口，
`hybrid_search()` 仍是自研的。

**二、时效过滤。** 条款要判断"在查询基准日是否现行有效"，涉及 `status` 字段与
`effective_date` 的两段式比对。LangChain 的过滤参数是字符串表达式，表达不了这个逻辑，
也不该由它表达——这是业务规则，不是检索策略。

**三、引用校验。** 回答生成后要逐条核对数字（金额、期限、比例）是否与条款原文一致，
不一致就降级。这一步**不在生成链路上，是生成之后的事后审计**，LangChain 没有对应落点。

### 中文切分踩过的坑

`RecursiveCharacterTextSplitter` 默认分隔符是英文标点（`\n\n`、` `、`.`）。
直接用于中文制度文本会在句子中间断开。改法是重写 `separators` 为
`["\n\n", "\n", "。|！|？", "；", "，"]` 并开 `keep_separator=True`，
同时覆盖 `_split_text_with_regex`：默认实现会把分隔符拼到片段开头，
中文需要在句末保留标点，否则带标点的句子会被切成两段。

实测：三块首字均为正文首字、末字全部为 `。`；4 份制度共 54 条 / 54 个切片，
子块长度 60~210 字（上限 300），无一篇超限。

## 两个关键实现细节

**父块严格等于一个"条"。** 检索按子块算相似度，送给模型的是整条原文。
这样引用的条号永远不会串，也不会出现"依据前半条作答、结论来自后半条"的错位。
过长的条款按（一）（二）枚举项拆成多个父块，仍不跨条。

**效力判断做两次。** 向量库侧按 `status` 字段过滤一次，取回结构化库后再按
施行日期与查询基准日比对一次。前者避免废止条款占据召回名额，后者拦住
"已发文但尚未施行"的制度被当成现行制度引用。

## FAQ 层：为什么它比检索更危险

高频标准问法走 `faq_standard` 表，BM25 匹配（jieba 分词）命中后**绕过检索直接返回**
预审定的答案，省掉一次多路召回加重排。但这条捷径的风险比完整链路大得多，
所以在设计上加了四道闸：

| 闸门 | 作用 | 不加会怎样 |
| --- | --- | --- |
| `clause_keys` 回表校验 | 答案引用的条款必须仍存在且为 `active`，任一不满足就整条放弃、降级走完整链路 | 条款废止后 FAQ 还在直出旧结论 |
| `last_verified` 复核期 | 超过 `FAQ_VERIFY_TTL_DAYS` 未人工复核的条目不直接返回 | 两年前审定的答案措辞越确定、用户越容易信，但可能已跟不上现状 |
| `invalid_at` 软失效 | 不物理删除，保留失效记录与原因 | 事后无法解释"为什么这个问题曾经这样答" |
| 锚点摘句 | 答案必须是含主干的完整句，不能是脱离上下文的列表项 | 直出"（二）股票、债券……；"这种没有主语的半句 |

前三道闸的共同逻辑：**FAQ 命中绕过检索，等于放弃了实时校验，就必须用入口校验补回来。**
判定 FAQ 是否可用不只看字符串像不像，还要看它指向的条款今天是否还成立。

第四道闸是本轮踩出来的坑，详见下一节。

## 踩坑记录

这部分是实际调试中撞到并修掉的问题，按"现象 → 根因 → 改法"记。

### 1. FAQ 答案只摘到半句列表项

**现象**：问"贷款资金可以拿去炒股吗"，FAQ 直出答案是
`（二）股票、债券、期货、金融衍生品和资产管理产品投资；`——一句没有主语的半句。

**根因**：制度原文的分项列表靠**换行**分段：

```
个人贷款用途应当符合法律法规规定和国家有关政策，不得用于下列用途：

（一）购房首付款、房地产开发相关支出；

（二）股票、债券、期货、金融衍生品和资产管理产品投资；
```

摘句用的切分正则只认句号和分号 `(?<=[。；])`，而引出分项的主干以**冒号**结尾，
于是主干和分项被切成两块，锚点落在分项上就只摘出了分项本身。

**改法**：切分正则加入换行 `(?<=[。；])|\n+`，并在锚点命中分项时向前回溯：
只要当前段是以「（一）」这类序号开头的分项，就继续往前找，直到撞上非分项的主干为止。
回溯条件不能写成"检查上一段是否以冒号结尾"——锚点是「（二）」时，「（一）」横在
中间、不以冒号结尾，回退一次就停了。这个边界是实测对比多种回溯条件后才定下来的。

**这件事的面试价值**：它说明「FAQ 绕过检索直出答案」这个设计的真实风险——
**答案一旦脱离上下文就会变成错误的答案**，而不是"缩短了但意思没变"。

### 2. `citation` 是 property，不是数据库字段

往 `Clause` 里塞 `citation=row["citation"]` 直接 `KeyError`。`Clause.citation` 是
`@property`，由 `title + doc_no + clause_no` 现拼；`status` 是必填字段但容易漏。
教训：**把 ORM row 转 dataclass 时，字段清单要以 dataclass 定义为准，不能凭 SQL 列名猜。**

### 3. RAGAS 0.4 与项目 API 不兼容

装最新 `ragas 0.4.3` 后启动即崩：`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`。
原因是 ragas 在 `llms/base.py` 里硬 import 了 `ChatVertexAI`，而
`langchain-community 0.4.x` 已把该模块迁走——这是 ragas 自身的 bug。
更重要的是 0.4 把指标 API 全改了。**降到 0.2.15** 并把评估环境隔离到 `.venv-ragas`。

### 4. 缺 API Key 的检查被 RAGAS 吞掉

最初把 key 检查放在 `make_judge()` 里 `raise SystemExit`，结果被 ragas 的
`evaluate` 内部捕获降级成一条警告，程序继续跑完并打出一堆毫无意义的分数。
**改法**：独立出 `require_api_key()`，在 `main()` 里、任何 `evaluate` 之前调用。

### 5. 余额耗尽时 RAGAS 输出 nan 而不报错

DeepSeek 返回 `402 Insufficient Balance`，28 次裁判调用全部失败，但 `evaluate`
仍正常结束，报告里四个指标全是 `nan`。**这类 nan 必须在 README 和 CI 里明确拦截**，
否则会被误当成"跑通了"。判断依据：报告出现 `nan` 即视为本轮评估无效。

## 可以优化的地方

按投入产出排序，前三条是"如果还有一天时间我会先做"的：

**① FAQ 层补 embedding 相似度做粗筛。** 当前只用 BM25 分词匹配，同义改写的问法
（"能贷多少" vs "额度上限"）靠 BM25 命中不了。改动很小：复用已有的 BGE-M3
给标准问法预编码，先向量召回 top-10、再 BM25 精排，成本约半天。

**② `last_verified` 自动巡检做成定时任务。** 现在 `check_faq_validity()` 要手动跑。
接成每日定时任务，过期条目自动降级 + 告警，才真正形成"FAQ 不会腐坏"的闭环。

**③ 评估集扩到 100+ 条。** 32 条已经能把四个指标稳定测出来，但按条线分层后
每条线只有 8 题，单题波动对分层结论影响偏大。

**④ 引用校验从"数字比对"扩展到"数值区间"。** 现在只校验 `\d+ + 单位` 是否在原文出现，
抓不到"50万写成 60万"以外的错误（如单位换算、口径混淆）。

**⑤ 增量入库。** 当前 `--drop` 全量重建。制度改一条要重建全库，规模上去后不可接受。

## 环境复现

主环境与 RAGAS 环境分开，原因是依赖冲突不可调和：

| 包 | 主环境 | .venv-ragas |
| --- | --- | --- |
| langchain | 1.4.7 | 0.3.30 |
| langchain-core | 1.7.0 | 0.3.86 |
| langchain-openai | 1.7.3 | 0.2.14 |
| ragas | — | 0.2.15 |
| sentence-transformers | — | 6.2.0 |

ragas 0.2.x 的 `llms/base.py` 里硬 import 了
`langchain_community.chat_models.vertexai.ChatVertexAI`，该校验在
community 0.4.x 之后已拆分到独立包，装上会直接 `ModuleNotFoundError`。
ragas 0.4 又整体改了 API，与本仓库按 0.2 写的评估脚本不兼容，
因此锁 0.2.15 并把整套 langchain 降到 0.3.x。

```bash
python -m venv .venv-ragas
.venv-ragas/Scripts/python.exe -m pip install \
    "ragas==0.2.15" "langchain==0.3.30" "langchain-core==0.3.86" \
    "langchain-community==0.3.31" "langchain-openai==0.2.14" \
    "langchain-huggingface==0.1.2" "sentence-transformers" datasets
```

安装后若有 `langchain-classic`、`langgraph` 系列残留，需卸载——
它们要求 `langchain-core>=1.4.7`，会把上面锁定的 0.3.86 顶掉。

评估用的 .env 变量：

```
POLICY_MYSQL_PASSWORD=...
POLICY_REDIS_PASSWORD=...
POLICY_LLM_API_KEY=sk-...        # RAGAS 评判模型与主链路共用
POLICY_EVAL_MODEL=deepseek-chat  # 可选，RAGAS 评判默认用这个
PYTHONIOENCODING=utf-8           # Windows 控制台必加，否则中文输出报 UnicodeEncodeError
```

## 免责说明

仓库内制度文本为演示用示例，不构成任何真实业务口径。
