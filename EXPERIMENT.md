# NOVA-F 实验记录

## 1. 项目目标

NOVA-F 面向 DataCon 2025 漏洞攻击流量识别任务，输入为 HTTP 请求流量，输出该请求对应的 CVE 标签列表。任务本质不是普通二分类，而是带有大量空标签、多标签、长尾 CVE 的多标签检索识别问题。

当前系统最终采用本地离线检索式架构：

```text
HTTP payload 清洗
-> SentenceTransformer 向量编码
-> FAISS top-k 近邻检索
-> CVE 候选聚合
-> 空标签近邻负证据抑制
-> 训练集 OOF blocklist
-> 可选 structured rerank
-> 可选结构化规则配置补召回
-> 输出 CVE labels
```

## 2. 初始 NOVA-F 的核心问题

### 2.1 工程环境不稳定

最初在 Windows 和 WSL 间切换运行时出现过多个环境问题：

```text
WSL 缺少 faiss
Python 版本与 faiss-cpu==1.8.0 不兼容
venv ensurepip 失败导致没有 pip
WSL 访问 HuggingFace 存在证书和代理问题
Windows 已下载模型但 WSL 不稳定联网
```

处理方式：

```text
使用 WSL conda 环境 nova-f
将 all-MiniLM-L6-v2 模型落地到 models/all-MiniLM-L6-v2
运行时显式指定 --model-path ./models/all-MiniLM-L6-v2
```

最终 WSL-Ubuntu 验证结果：

```text
Python 3.12.13
faiss/pandas/numpy import ok
main.py 全量流程可运行
```

### 2.2 数据路径和格式不统一

官方授权数据原始格式为：

```text
train.json.gz
test.json.gz
```

项目主流程需要 CSV，因此新增：

```text
utils/convert_datacon_jsonl.py
```

统一输出字段：

```text
id
payload_decoded
labeled
cve_labels
```

### 2.3 空标签解析 bug

早期 `normalize_cve_labels()` 对 pandas `NaN` 没有特殊处理，空标签可能被转成字符串 `"NAN"`，污染中间统计和标签判断。

修复位置：

```text
src/preprocess.py
```

修复逻辑：

```text
None / NaN / "nan" / "none" / "null" -> []
list 内空值同样过滤
```

### 2.4 初始算法的主要错误类型

早期系统使用扩展训练集 `train_with_ultimate.csv` 做向量检索，在官方授权测试集上表现为：

```text
all_rows
precision: 0.156417
recall:    0.757105
micro_f1:  0.259270
macro_f1:  0.479125
```

这个结果说明系统不是找不到 CVE，而是把大量空标签或非 CVE 流量误报成 CVE。瓶颈首先是 precision，而不是继续盲目扩大召回。

## 3. 优化工作流总览

整体优化遵循以下原则：

```text
先建立可复现评估
再定位主要错误来源
再做训练集可验证优化
最后区分真实场景路线和测试集实验上界
```

每轮实验都记录：

```text
precision
recall
micro_f1
macro_f1
per-label F1
zero-F1 CVE 数量
top FN / top FP
```

## 4. 实验分支一：combined 索引

### 问题

原始扩展训练集偏向 CVE 正样本，缺少官方分布中的空标签和负样本。检索系统只看到“相似攻击样本”，看不到“相似但不应标 CVE 的样本”。

### 方法

新增：

```text
utils/merge_training_csv.py
```

将：

```text
train_with_ultimate.csv
train_payload.csv
```

合并为 combined 训练空间，并构建：

```text
embeddings/faiss_store_combined
```

### 结果

仅使用 combined 索引后：

```text
base_threshold=0.84
min_votes=1
vote_weight=0.015

precision: 0.522833
recall:    0.751556
micro_f1:  0.616669
macro_f1:  0.483007
```

### 分析

这一步是第一轮大幅提升。原因是负样本进入检索空间后，系统能区分“相似但不应输出 CVE”的请求。

## 5. 实验分支二：CVE 候选聚合

### 问题

原始近邻策略接近“取第一个 CVE 标签”。这对噪声近邻敏感，也没有利用多个近邻同时支持同一 CVE 的证据。

### 方法

修改：

```text
src/search_faiss.py
```

新增候选聚合：

```text
score = best_similarity + min(votes - 1, 8) * vote_weight
```

参数：

```text
--min-votes
--vote-weight
```

### 结果和分析

`min_votes=2/3` 会伤害长尾 CVE，因为低频 CVE 本来就近邻少。最终保留：

```text
min_votes=1
vote_weight=0.015
```

这是一种低风险增强：保留单近邻召回能力，同时让重复证据略微加分。

## 6. 实验分支三：空标签近邻负证据

### 问题

combined 索引降低了误报，但仍有大量空样本被输出 CVE。需要利用空标签近邻作为负证据。

### 方法

修改：

```text
src/search_faiss.py
```

核心逻辑：

```text
如果最强空标签近邻与最强 CVE 近邻非常接近，
且空标签近邻数量达到比例要求，
则抑制 CVE 输出。
```

关键参数：

```text
empty_penalty_margin=0.05
empty_penalty_floor=0.80
empty_penalty_ratio=0.50
```

### 结果

combined + empty-neighbor suppression：

```text
precision: 0.666632
recall:    0.710354
micro_f1:  0.687799
macro_f1:  0.523245
```

### 分析

precision 大幅提升，recall 有一定下降。考虑初始瓶颈是误报，这个取舍是合理的。

## 7. 实验分支四：更强 embedding 模型

### 问题

尝试验证是否主要瓶颈来自表示模型不够强。

### 方法

测试：

```text
BAAI/bge-small-en-v1.5
intfloat/e5-small-v2
```

E5 使用：

```text
passage:
query:
```

### 结果

BGE small 在小 holdout 上只有轻微收益，但 CPU 成本更高；E5 small 低于 MiniLM。

### 分析

结论是暂不替换默认 embedding。当前瓶颈主要在决策层、负样本抑制、长尾覆盖和结构化特征，而不是单纯向量模型。

## 8. 实验分支五：训练集 OOF blocklist

### 问题

部分 CVE 在预测中系统性高 FP、低 precision。直接用官方测试集错误分析写 blocklist 有数据泄漏风险。

### 方法

新增：

```text
utils/learn_blocklist_from_folds.py
```

使用训练集 out-of-fold 预测统计稳定低 precision、高 FP 的 CVE：

```text
min_fp=20
max_precision=0.02
min_folds=2
```

输出：

```text
data/experiments/fold_blocklist_fp20_p002_mf2.txt
```

### 结果

OOF blocklist 后：

```text
precision: 0.758393
recall:    0.709120
micro_f1:  0.732930
macro_f1:  0.535373
```

### 分析

这是可辩护的真实场景优化，因为 blocklist 来源是训练集 OOF，而不是测试集真值。

## 9. 实验分支六：手写 signature rescue

### 问题

错误分析发现部分高 FN CVE 有明显路径特征，例如 `/wsman`、`/remote/fgt_lang`、`/onvif-http/snapshot`。

### 方法

曾实现手写规则实验，结果上界较高：

```text
precision: 0.772989
recall:    0.753966
micro_f1:  0.763359
macro_f1:  0.539183
```

### 分析

该分支不作为真实主线。原因是部分规则来自官方测试集错误分析，泛化证据不足。后续删除对应脚本，只在报告中保留为实验上界。

唯一保留思想是 `wsman-38649`：它在训练集中验证充分，可作为定向 group completion。

## 10. 实验分支七：自动 path 规则挖掘

### 问题

手写规则泛化风险高，因此需要从训练集自动挖掘高精度规则。

### 方法

曾实现 path-only 规则挖掘和 OOF 验证。结果显示 exact path rules 有较强训练集 OOF 证据：

```text
exact path, min_precision=1.0, min_support=20
OOF precision: 0.992277
OOF recall:    0.635369
OOF micro_f1:  0.774552
OOF macro_f1:  0.860057
```

应用到官方授权测试集：

```text
OOF blocklist + wsman-38649 + exact mined path rules
precision: 0.758397
recall:    0.735355
micro_f1:  0.746699
macro_f1:  0.538537
```

### 分析

该路线成为保守主线。后续 path-only 脚本被结构化规则工具替代，已删除旧脚本。

## 11. 实验分支八：Macro-F1 与长尾诊断

### 问题

即使 micro-F1 提高，macro-F1 仍明显偏低。

### 方法

新增：

```text
utils/evaluate_per_label.py
utils/diagnose_long_tail.py
```

诊断当前主线：

```text
labels: 1311
zero_f1_labels: 540
low_f1_labels_lt_0.2: 554
```

### 分析

Macro-F1 低的根因是长尾 CVE 大量 zero-F1。Micro-F1 被高频 CVE 主导；Macro-F1 对每个 CVE 等权，长尾未召回会显著拉低平均。

## 12. 实验分支九：全局多标签扩展

### 问题

部分请求真实存在多个 CVE，但系统可能只输出一个。尝试复制近邻 labelset 或 top-k consensus。

### 方法

曾测试：

```text
copy-top1
consensus
labelset completion
```

### 结果

最佳结果仍低于当前主线：

```text
copy-top1 best micro_f1: 0.746129
consensus best micro_f1: 0.746585
```

### 分析

全局多标签扩展会引入 FP，收益不稳定。该分支失败，脚本已删除。多标签修复只保留训练验证充分的定向规则，例如 `wsman-38649`。

## 13. 实验分支十：结构化签名规则

### 问题

path-only 规则覆盖有限，很多漏洞特征在 query key、body key、参数值或特殊 token 中。

### 方法

新增：

```text
src/structured_features.py
utils/structured_signature_rules.py
utils/export_rule_config.py
utils/apply_rule_config.py
```

结构化签名类型：

```text
path
query_keys
body_keys
query_key_value
body_key_value
token
```

规则配置字段：

```text
rule_id
enabled
rule_type
signature
target_cve
support
precision
source
risk_level
```

### 结果

结构化规则 OOF：

```text
min_support=20, min_precision=1.0
precision: 0.991996
recall:    0.622656
micro_f1:  0.764982
macro_f1:  0.848935
```

低 support 对比：

```text
support=10 micro_f1=0.714005
support=5  micro_f1=0.724278
```

### 分析

低支持规则数量更多但泛化下降，因此最终采用 support=20。规则配置化解决了硬编码和审计问题。

## 14. 实验分支十一：structured rerank

### 问题

纯 embedding 相似度不能充分利用 HTTP 结构信息。两个 payload 语义向量接近，但 path/query/body 结构可能不同。

### 方法

新增：

```text
utils/structured_rerank_experiment.py
main.py --structured-rerank-alpha
main.py --train-feature-path
```

结构化加权：

```text
score = semantic_score + alpha * feature_bonus
```

`feature_bonus` 包含：

```text
method match
path exact match
path token Jaccard
query key Jaccard
body key Jaccard
payload token Jaccard
```

### 结果

单独 rerank：

```text
alpha=0.000 micro_f1=0.687127 macro_f1=0.524130
alpha=0.030 micro_f1=0.693526 macro_f1=0.532527
```

组合召回增强路线：

```text
structured rerank + OOF blocklist + rule-config structured rules only-empty + wsman-38649
precision: 0.739922
recall:    0.756264
micro_f1:  0.748004
macro_f1:  0.548448
```

### 分析

structured rerank 提高 recall 和 macro-F1，但牺牲 precision。因此它不替代默认主线，而作为 recall-first 模式。

## 15. 实验分支十二：条件式 blocklist

### 问题

全局 CVE blocklist 可能误伤真实攻击。希望如果 payload 命中该 CVE 的 allow signature，则保留。

### 方法

新增：

```text
utils/conditional_filter_predictions.py
```

### 结果

```text
Preserved by allow signatures: 0
micro_f1: 0.738662
macro_f1: 0.546500
```

### 分析

工具完成，但当前 allow config 没有覆盖 blocklist 内误伤项，因此暂不采纳。

## 16. 数据缺口分析

### 方法

新增：

```text
utils/export_data_gap_report.py
```

导出无训练支持且 FN>=20 的 CVE。

### 结果

当前数据缺口 10 个：

```text
CVE-2021-20016
CVE-2024-1800
CVE-2020-8949
CVE-2017-17215
CVE-2020-35391
CVE-2017-6514
CVE-2023-3306
CVE-2023-26801
CVE-2013-3307
CVE-2021-43163
```

### 分析

这些 CVE 不能靠训练集可验证规则安全补齐。真实提升需要补授权样本或公开样本；否则硬写规则就是测试集过拟合。

## 17. 最终系统工程架构

### 17.1 主要模块

```text
main.py
  一站式清洗、建库、检索、预测入口

src/preprocess.py
  HTTP payload 清洗、CVE 标签归一化

src/build_faiss.py
  SentenceTransformer 编码与 FAISS 索引构建

src/search_faiss.py
  向量检索、候选聚合、空标签负证据、blocklist、structured rerank 接入

src/structured_features.py
  HTTP method/path/query/body/token 结构化特征

utils/evaluate_predictions.py
  micro/macro/precision/recall 评估

utils/evaluate_per_label.py
  per-CVE 指标与 zero-F1 诊断

utils/learn_blocklist_from_folds.py
  训练集 OOF blocklist 学习

utils/structured_signature_rules.py
  结构化规则挖掘、OOF 验证、应用

utils/export_rule_config.py / apply_rule_config.py
  规则配置化和审计化

utils/export_data_gap_report.py
  无训练支持 CVE 缺口导出
```

### 17.2 算法核心

核心算法不是端到端分类，而是检索增强的弱监督多标签识别：

```text
1. 清洗 HTTP 报文，删除弱区分度请求头，标准化空白。
2. 使用 all-MiniLM-L6-v2 对 payload_clean 编码。
3. FAISS IndexFlatIP 做 top-k 内积检索。
4. 对近邻 CVE 做聚合：
   best_similarity + vote bonus
5. 使用 adaptive threshold 输出 1~3 个候选 CVE。
6. 如果强空标签近邻接近 CVE 近邻，则抑制输出。
7. 可选 OOF blocklist 删除训练集统计出的系统性误报 CVE。
8. 可选 structured rerank 提高 path/query/body 结构一致样本的近邻得分。
9. 可选结构化规则配置对空预测补召回。
```

## 18. 最终主线

### 18.1 precision-first 默认主线

真实场景默认建议：

```text
OOF blocklist + wsman-38649 + exact/structured high-precision rules

precision: 0.758397
recall:    0.735355
micro_f1:  0.746699
macro_f1:  0.538537
```

选择原因：

```text
泛化证据最强
误报更低
适合无人值守或高误报代价场景
```

### 18.2 recall-first 候选路线

如果更重视召回和 Macro-F1：

```text
structured rerank + OOF blocklist + rule-config structured rules only-empty + wsman-38649

precision: 0.739922
recall:    0.756264
micro_f1:  0.748004
macro_f1:  0.548448
```

选择原因：

```text
micro-F1 略高
macro-F1 更高
长尾覆盖略好
适合后续有人审告警的场景
```

### 18.3 实验上界

```text
hand-written signature rescue
micro_f1: 0.763359
```

不作为真实主线，因为有测试集错误分析痕迹。

## 19. WSL-Ubuntu 完整运行验证

WSL 环境：

```text
Python 3.12.13
conda env: nova-f
faiss/pandas/numpy import ok
```

200 行 smoke：

```text
main.py structured rerank smoke passed
```

全量运行：

```text
python main.py \
  --train-path ./data/train_with_ultimate.csv \
  --test-path ./data/test_payload.csv \
  --test-payload-column payload_decoded \
  --store-dir ./embeddings/faiss_store_combined \
  --output-path ./data/experiments/wsl_full_recall_first.csv \
  --model-path ./models/all-MiniLM-L6-v2 \
  --reuse-cache \
  --structured-rerank-alpha 0.03 \
  --train-feature-path ./data/experiments/train_combined_cleaned.csv \
  --prediction-blocklist ./data/experiments/fold_blocklist_fp20_p002_mf2.txt
```

运行结果：

```text
rows: 105077
prediction file: data/experiments/wsl_full_recall_first.csv
```

该全量主流程输出为 rerank+blocklist 阶段结果：

```text
precision: 0.735511
recall:    0.737037
micro_f1:  0.736273
macro_f1:  0.549538
```

完整 recall-first 最优结果需要在该类输出后继续应用规则配置和 `wsman-38649` 等已验证后处理。

## 20. 当前未解决问题

当前代码侧主要优化已收敛，剩余问题主要是数据覆盖：

```text
zero-F1 CVE 仍多
部分高 FN CVE 训练集中没有样本
无训练支持 CVE 不应硬写规则
```

下一步真正提高上限，需要补充授权样本或公开样本，然后重建索引和规则。

