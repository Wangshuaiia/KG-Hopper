> 本文是对**修改前**代码（commit 195c47f）的审查记录，其中的问题已在后续修改中处理，见 README 的"改动说明"。

# KG-Hopper 代码审查：与论文（arXiv:2603.21440）的对照

标注说明：🔴 阻断（跑不起来） · 🟠 逻辑错误（能跑但结果错） · 🟡 与论文不一致 · ✅ 已用 `sim/` 实测复现

## 0. 论文流程 vs 代码现状

```
论文:  冷启动 SFT(500 条, 三元组 mask) → GRPO(2000 条, 16 rollouts)
          rollout: <think> … <search>实体</search> → KG 两阶段检索 → <searched_triples>…</searched_triples> … <answer>
          reward : R_search + R_format + R_reason(70B 裁判) + R_answer(3B 裁判)
          history resampling: 第 2 个 epoch 起去掉一跳问题
       → 8 个数据集评测 Hits@1

代码:  [无 SFT 代码/数据]
       训练 rollout(OpenRLHF-RAG, 从 R1-Searcher 改来): <|begin_of_query|>关键词<|end_of_query|>
          → POST http://0.0.0.0:5003/queries（维基百科文档检索服务，仓库里没有）→ <|begin_of_documents|>
       reward_server.py: F1 + 答案裁判 + 推理裁判, 格式错误 -2
       评测 eval_search_kg.py: <search>/<searched_triples> + kg-tool(SPARQL, 依赖缺失)
```

结论：**训练端完全没有接入知识图谱**，训练和评测使用两套不同的标签和检索器。

## 1. 🔴 阻断问题

| # | 位置 | 问题 |
|---|---|---|
| B1 | `kg-tool/utils.py:12`, `kg-tool/kg_search_function.py:8,10,96` | 导入了不存在的模块 `config`、`gnn_base_train`、`freebase_func`。✅ 去掉第三方依赖后仍报 `No module named 'config'` |
| B2 | `kg-tool/kg_search_function.py:14-23` | import 时就加载一个 GNN 模型（`MODEL_CONFIG["model_state_path"]`），论文里没有 GNN，权重也不在仓库 |
| B3 | `kg-tool/kg_search_function.py:350-370` `wiki_retrieval` | ① 使用未定义的全局变量 `topic_entity`；② 调 `relation_search_prune`/`entity_score` 时缺少必需参数 `args`；③ 返回 tuple，评测里 `'\n'.join(...)` 会 TypeError；④ tuple 顺序是 `(relation, head_MID, tail)`，MID 会直接暴露给模型；⑤ 评测传入的是实体**名称**，SPARQL 需要 **MID**，缺少名称→MID 的映射 |
| B4 | `evaluation/eval_search_kg.py:14` | `from .extract_entity_from_query import …` 是相对导入，按 README 的 `python evaluation/eval_search_kg.py` 运行会失败 |
| B5 | `evaluation/eval_search_kg.py:137` | `data_path_qs_list = []`，什么都不评测 |
| B6 | `OpenRLHF-RAG/.../experience_maker.py:1058` | rollout 写死调用 `http://0.0.0.0:5003/queries`（维基检索服务），仓库里没有这个服务，也不是 KG |
| B7 | `reward-remote/reward_server.py:22,30` | 裁判模型路径是占位符；README 写的文件名 `reward_server_qwen_zero.py` 与实际不符，启动命令也缺少必需参数 `--data_path --reward_pretrain` |
| B8 | `reward_server.py:219` + `prompts_dataset.py:8,34` | 需要 `datasets.load_from_disk` 格式的训练集，字段为 `question/answer/idx`；仓库里没有训练数据，也没有构造脚本 |
| B9 | — | 没有冷启动 SFT 的脚本和数据（论文 III-B）；`sft_dataset.py` 也不支持对三元组做 loss mask |
| B10 | `requirements.txt` | 缺少 SPARQLWrapper、rank_bm25、sentence-transformers、openai、fastapi、uvicorn、ujson、ray、requests |
| B11 | `evaluation/extract_entity_from_query.py:40` | ⚠️ **公开仓库里硬编码了 API key 和公司内网代理地址** |

## 2. 🟠 逻辑错误

（模拟证据见第 4 节）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| L1 ✅ | `eval_search_kg.py:120` | 检索后只拼接了 `</search>` 和三元组，**没有拼接模型本轮生成的文本**（推理和 `<search>查询`） | 每轮检索后模型都丢失之前的推理；最终文本里 `<search>` 和 `</search>` 数量不匹配 |
| L2 | `reward_server.py:108-115` | `decode(generation_output[0])` 包含整个 prompt，而 prompt 里先出现了 `return '1'`，所以 `re.search(r"\b([01])\b")` **永远匹配到 1** | R_answer 恒为 1，答错也拿满分 |
| L3 | `reward_server.py:75-78` | 推理裁判同样解码了 prompt；`truncation=True, max_length=2048` 从右侧截断，长轨迹的结尾（结论）会被截掉；70B 模型用 HF fp16 放在单卡上放不下 | R_reason 不可靠 |
| L4 | `reward_server.py:284-295` | 格式检查要求 `<searched_triples>\n(1)` 的次数与其他标签相等，但 KG 工具返回的三元组没有 `(1)` 编号 | 每条带检索的轨迹都被罚 -2 |
| L5 | `reward_server.py:328` | 正则写成 `<searched_triples>>`（多了一个 `>`），三元组没被剔除，三元组里的中文会触发"含中文"惩罚 | |
| L6 | `reward_server.py:319,325` | 遗留了 `begin_of_query` 检查；答案超过 10 个词就罚 -2，多答案问题（WebQSP 平均多个答案）会被误罚 | |
| L7 ✅ | `experience_maker.py:1266-1271` | 检索内容的 mask 用硬编码 token ID 匹配。✅ 用真实 tokenizer 验证：只对应 `<\|begin_of_documents\|>`，换成论文的 `<searched_triples>` 后完全匹配不上，而且只 `print` 一条 "Important Bug" 就继续 | 三元组会参与 loss，违背论文 III-D |
| L8 | `eval_search_kg.py:99-130` | 10 轮内没有作答的样本**不写任何结果**，抛异常的样本也被 `except` 吞掉 | 分母变小，指标虚高 |
| L9 ✅ | `eval_search_kg.py:45-48` | slot_filling 和 fact_checking 的 prompt 正文只有 `...` | T-REx/Zero-Shot RE/Creak 的评测无效 |
| L10 | `eval_search_kg.py:53` | `json.load(f)[14:]` 跳过每个数据集的前 14 条 | |
| L11 | `eval_search_kg.py:85` | gold 存的是 `str(...)`：WebQSP 是空串（答案在 `Parses[].Answers`），GrailQA 是 dict 的字符串；也没有计算 Hits@1 的代码 | |
| L12 | `experience_maker.py:1207` | 续写时用 `generated_text.strip()` 拼接后重新分词，训练用的 token 序列和实际采样出的不完全一致（轻微的 off-policy） | |
| L13 | 训练 prompt vs 评测 prompt | 训练要求"关键词"查询（`prompts_dataset.py:27`），评测要求"只写实体"（`eval_search_kg.py:43`） | 训练/评测分布不一致 |

## 3. 🟡 与论文不一致

| 论文 | 代码 |
|---|---|
| GRPO（III-D） | `--advantage_estimator reinforce_baseline`（REINFORCE++ baseline，不除以 std）；`group_norm` 才是 GRPO 的组内归一化 |
| KL 系数 1e-5 | `--init_kl_coef 0.0` |
| lr 1e-6, batch 16, 2 epochs | lr 2e-6, TBS 256 / RBS 64, `max_epochs 1`, `num_episodes 10000` |
| R_search = min(0.5n, 0.8) | 没有 |
| R_format = +0.5 | 格式错误时 -2；没有 `<answer>` 时总分直接置 0 |
| R_reason ∈ (0,1), R_answer ∈ {0,1} | 还额外加了一项 F1（总分可达 3） |
| history resampling（第 2 个 epoch 起去掉一跳问题） | 没有实现 |
| 冷启动 SFT 500 条 | 没有 |
| 两阶段 KG 检索：取全部出边 → 选最相关谓词 | kg-tool 里还混着子问题生成、LLM 关系打分、BM25/SentenceTransformer/GNN 剪枝，大多来自 ToG 一类的旧代码 |
| 每次检索 → 三元组 | 训练 rollout：维基文档 top-3；每轮 `max_tokens=512` 写死，最多 8 次检索，`--temperature` 和 `--generate_max_len` 都不生效 |

## 4. 模拟实测（`sim/`）

环境：
- 策略模型：本地 Qwen2.5-1.5B-Instruct（MPS）
- 知识图谱：Claude 模拟 Freebase（`claude-opus-5-5`, effort=low），两阶段检索
- 原始 `eval_search_kg.main()` 和 `reward_server.get_reward()` 原样执行，只打了 P1（去掉 `[14:]`）和 P2（相对导入）两个补丁

### 4.1 评测流程（`run_eval_sim.py`）

**zero-shot**：`--n 3`，cwq / WebQuestions / T-REx / Creak，共 12 条
- 未经训练的模型几乎不检索，直接作答，这与论文"需要冷启动 SFT"一致
- ✅ **L9**：T-REx 和 Creak 共 6 条全部 `shot_down`。prompt 里只有 `...`，模型输出 `[epic fantasy]`、`<verify>…`、`<searched_sources>` 等五花八门的格式

**few-shot**：`SIM_FEWSHOT=1`，用一条格式示范模拟冷启动后的模型，cwq / WebQuestions 各 3 条
- 模型会按格式发起检索。模拟知识图谱返回了合理的三元组，例如 `(Lou Seal, sports.mascot.team, San Francisco Giants)`
- ✅ **L1**：6 条最终文本都是 `<search>`=0、`</search>`=1。模型本轮写的"先查 Lou Seal 属于哪个队… `<search> Lou Seal`"被整段丢掉，下一轮 prompt 直接从 `</search>\n\n<searched_triples>` 接上
  - 后果：拿到第一跳结果后，模型不记得自己的计划，**没有一条完成第二跳检索**，只能凭参数记忆作答（Lou Seal 那题答成 "2021"，gold 是 2014）
  - 论文的核心卖点"单轮推理内的多跳检索"在现有评测代码里根本无法发生

### 4.2 奖励函数（`run_reward_sim.py`）

- ✅ **L2**：`evaluate_answer(pred='Paris', gold='Tokyo')` 返回 **1.0**，而裁判模型实际生成的是 `'0\n\nThe predicted answer'`。所有轨迹的 R_answer 都是 1，不管答案对错
- ✅ **L4**：答案正确、格式完美、两跳检索的轨迹，只因三元组没有 `(1)` 编号，就被罚 -2（总分从约 3 跌到约 1）
- ✅ **奖励的激励方向是反的**：
  - "不检索、直接答对"的得分 ≥ "两跳检索后答对"，因为代码里没有 R_search
  - KG 工具的真实输出（不编号）还会被罚分
  - RL 会把模型推向**不检索**，与论文目标相反
- ✅ **答错也有高分**：答 "2010 World Series"（gold 2014）时，F1 部分得分 0.67 + R_answer 恒为 1，总分接近答对
- ✅ **训练 rollout 产生的 R1-Searcher 标签不受任何格式约束**：格式检查只统计 `<search>`/`<searched_triples>`，`<|begin_of_query|>` 轨迹被当成"没检索"，不罚分，检索次数也不计
- ✅ **L3**：推理裁判的打分解析很脆弱。裁判写 `The score for this reasoning process is **1**.` 时 R_reason 被解析成 0
  - 裁判按 `generation_config` 默认 `do_sample=True` 采样，同一条轨迹两次打分可能是 0 和 1，奖励噪声很大，应改成贪心解码并只解码新 token

实测得分（`get_reward` 原样执行；为了让分解精确，同一条轨迹的裁判分数只取一次）：

| 轨迹 | 答案 | 检索 | F1 | R_ans | R_reason | **代码总分** | 论文应得 R_answer |
|---|---|---|---|---|---|---|---|
| 参考：论文标签，三元组不编号（kg-tool 实际输出） | 对 | 2 跳 | 1.00 | 1 | 0.00 | **0.00**（-2） | 1 |
| 参考：论文标签，三元组编号 | 对 | 2 跳 | 1.00 | 1 | 0.00 | **2.00** | 1 |
| 参考：答 2010 World Series | **错** | 2 跳 | 0.67 | 1 | 1.00 | **2.67** | 0 |
| 参考：答 Boston Red Sox | **错** | 2 跳 | 0.00 | 1 | 0.00 | **1.00** | 0 |
| 参考：R1-Searcher 标签（训练 rollout 实际产出） | 对 | 1 次 | 1.00 | 1 | 0.50 | **2.50** | 1 |
| 参考：不检索 | 对 | 0 | 1.00 | 1 | 0.00 | **2.00** | 1 |
| zero-shot：Lou Seal → "The Chicago Cubs" | **错** | 0 | 0.00 | 1 | 0.90 | **1.90** | 0 |
| few-shot：Ovadia Yosef → "Judaism" | 对 | 1 | 1.00 | 1 | 0.00 | **0.00**（-2，L1+L4） | 1 |
| few-shot：其余 4 条（都发起了检索） | 错 | 1 | — | 1 | — | **-1.00 ~ 0.37**（都被 -2） | 0 |

> 答错的 2.67 > 答对的 2.00；不检索、答错的 1.90 > 检索后答对的 0.00。用这套奖励做 RL，模型会学到"别检索、随便答"。

（完整输出用 `python sim/run_reward_sim.py` 复现；裁判是 1.5B 小模型，所以 R_reason 数值本身不可信，但上面这些都是代码逻辑问题，换成 70B 裁判也同样存在）

## 5. 建议的修复顺序

1. **先定一套协议并全仓统一**：采用论文的 `<search>实体</search>` 和 `<searched_triples>\n(1)… (2)…</searched_triples>`。训练 prompt、rollout、mask token、奖励格式检查、评测 prompt 全部使用这一套
2. **重写 KG 工具为独立的 HTTP 服务**，接口为 `entity, question → triples`：
   - 实现论文的两阶段 SPARQL 检索，加上名称→MID 链接
   - 删除 GNN、子问题生成等无关代码
   - 训练 rollout 和评测共用这一个服务（`sim/mock_kg.py` 的接口可以直接作为它的模拟后端）
3. **修 rollout**（`experience_maker.py`）：
   - 改成 `<search>` 停止词并调用 KG 服务
   - mask 改为按字符区间映射到 token，不再硬编码 ID
   - 读取 `--temperature` 和 `--generate_max_len`
4. **按论文重写奖励函数**：
   - R_search + R_format(+0.5) + R_reason + R_answer
   - 裁判只解码新 token、贪心解码
   - 70B 推理裁判用 vLLM 起 OpenAI 兼容服务，而不是 HF 单卡加载
5. **修评测**：
   - L1（拼接 generated_text）、L8（未作答的样本也要记录）、L9（补全 prompt）、L10、L11
   - 增加 Hits@1 / Accuracy 计算脚本
6. **补齐论文流程**：
   - 冷启动 SFT 的数据生成（用开源大模型）、训练脚本、三元组 mask
   - 训练数据构造（2000 条，带 `idx`）
   - GRPO 超参：`group_norm`、kl 1e-5、lr 1e-6……
   - history resampling
7. **安全**：删除硬编码的 key 并**到服务端吊销**；仅从 git 历史里删除并不够，key 已经公开
