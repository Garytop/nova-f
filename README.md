# NOVA-F

NOVA-F 是一个面向 DataCon 2025 漏洞攻击流量识别任务的本地离线识别系统。输入 HTTP 请求流量，输出对应的 CVE 标签。

当前系统不是端到端训练分类器，而是检索式识别系统：

```text
HTTP payload 清洗
-> SentenceTransformer 向量编码
-> FAISS 近邻检索
-> CVE 候选聚合
-> 空标签近邻负证据抑制
-> OOF blocklist
-> 可选 structured rerank
-> 可选结构化规则补召回
```

## 当前状态

项目已经完成主要工程化优化，并在 WSL-Ubuntu 下跑通过全量主流程。

全量主流程 rerank+blocklist 阶段评估：

```text
precision: 0.735511
recall:    0.737037
micro_f1:  0.736273
macro_f1:  0.549538
```

当前推荐保留两种使用模式。

## 推荐模式

### precision-first

默认推荐主线。误报更低，泛化证据最强。

```text
OOF blocklist + wsman-38649 + exact/structured high-precision rules

precision: 0.758397
recall:    0.735355
micro_f1:  0.746699
macro_f1:  0.538537
```

适合误报代价较高的场景。

### recall-first

召回优先路线。提高 recall 和 Macro-F1，但 precision 会下降。

```text
structured rerank + OOF blocklist + rule-config structured rules only-empty + wsman-38649

precision: 0.739922
recall:    0.756264
micro_f1:  0.748004
macro_f1:  0.548448
```

适合后面还有人工复核的安全告警场景。

## 和初始版本对比

初始版本主要依赖扩展训练集做 embedding 检索，误报严重：

```text
precision: 0.156417
recall:    0.757105
micro_f1:  0.259270
macro_f1:  0.479125
```

当前保守主线：

```text
precision: 0.758397
recall:    0.735355
micro_f1:  0.746699
macro_f1:  0.538537
```

主要提升来自：

```text
combined 正负样本索引
CVE 候选聚合
空标签近邻负证据抑制
训练集 OOF blocklist
训练验证的结构化规则
可选 structured rerank
```

## Linux / WSL 环境配置

推荐直接使用安装脚本：

```bash
source ./setup_install.sh
```

安装完成后，可以用启动脚本运行：

```bash
./run.sh --help
./run.sh --mode precision
./run.sh --mode recall --alpha 0.03
```

脚本会：

```text
检查 Python 版本
创建 .venv
升级 pip/setuptools/wheel
安装 numpy、pandas、tqdm、faiss-cpu、sentence-transformers
验证核心依赖可导入
激活虚拟环境
```

如果不希望当前 shell 自动激活环境，也可以执行：

```bash
bash ./setup_install.sh
source ./.venv/bin/activate
```

如果 WSL 访问 HuggingFace 不稳定，建议先在 Windows 下载模型，然后放到：

```text
models/all-MiniLM-L6-v2
```

运行时显式指定：

```bash
--model-path ./models/all-MiniLM-L6-v2
```

## 数据准备

官方授权数据不进入 Git。

约定原始数据位置：

```text
data/datacon2025/datacon2025-xlab-httpcve/data-release/train.json.gz
data/datacon2025/datacon2025-xlab-httpcve/data-release/test.json.gz
```

转换为项目 CSV：

```bash
python utils/convert_datacon_jsonl.py \
  --input ./data/datacon2025/datacon2025-xlab-httpcve/data-release/train.json.gz \
  --output ./data/train_payload.csv

python utils/convert_datacon_jsonl.py \
  --input ./data/datacon2025/datacon2025-xlab-httpcve/data-release/test.json.gz \
  --output ./data/test_payload.csv
```

CSV 字段：

```text
id
payload_decoded
labeled
cve_labels
```

## 框架

项目整体遵循“数据清洗 -> 向量索引构建 -> 本地相似度检索 -> 自适应标签输出 -> 可选后处理增强”的流水线架构。

### 数据预处理

`src/preprocess.py` 提供 `clean_payload_text`、`preprocess_dataframe`、`normalize_cve_labels` 等函数。

`clean_payload_text` 对 HTTP 报文按统一规则处理：

```text
去除低区分度请求头
统一换行符
删除空行
标准化连续空白
保留 method、path、query、body 等高价值内容
```

训练集、测试集都通过 `main.py` 的入口函数生成带 `payload_clean` 列的 CSV：

```text
preprocess_training_data
preprocess_test_data
```

这样训练和测试使用同一套清洗规则，避免输入分布不一致。

`normalize_cve_labels` 负责解析 CVE 标签，兼容空值、字符串、Python list 字符串、JSON list 字符串、空格分隔标签。最终统一为去重、升序、大写的 `CVE-XXXX` 标签列表。

### 向量化与索引

默认使用本地 SentenceTransformer：

```text
models/all-MiniLM-L6-v2
```

也可以通过参数替换：

```bash
--model-path
--device
--train-text-prefix
--test-text-prefix
```

`src/build_faiss.py` 负责读取清洗后的训练集，批量编码 `payload_clean`，并生成：

```text
train_embeddings.npy
faiss.index
meta.json
```

索引使用 FAISS `IndexFlatIP`。向量会做 L2 归一化，因此内积相似度等价于余弦相似度。

`meta.json` 记录训练样本 ID、归一化标签列表、模型名称、向量文件名、索引文件名、时间戳等信息，供后续检索使用。`main.py` 的 `build_vector_store` 封装了这一流程。

### 离线检索与预测

`src/search_faiss.py` 负责检索和预测核心逻辑。

测试集向量会缓存到：

```text
test_embeddings.npy
test_cache_meta.json
```

缓存会校验测试集 ID、embedding 模型、`test_text_prefix`。匹配时直接复用，避免重复编码。

检索阶段调用：

```python
faiss.Index.search
```

可配置参数：

```bash
--top-k
--max-candidates
--search-batch-size
```

### CVE 候选聚合

原始版本接近“取最相似近邻的第一个 CVE”，容易受单个噪声近邻影响。当前改为对 top-k 近邻中的 CVE 证据做聚合：

```text
score = best_similarity + min(votes - 1, 8) * vote_weight
```

默认：

```text
min_votes=1
vote_weight=0.015
```

这样既保留长尾 CVE 的单近邻召回能力，又能利用多个近邻支持同一 CVE 的一致性。

### 自适应标签输出

`adaptive_predict` 根据多层阈值决定最终输出 0 到 3 个 CVE：

```text
base_threshold
high_confidence
medium_confidence
second_gap
third_gap
```

逻辑上先判断 top-1 是否超过基础阈值，再根据第二、第三候选的置信度和分差决定是否补充多标签。

### 空标签近邻负证据

官方训练数据中存在大量空标签或非 CVE 样本。当前系统把这些样本作为负证据。

如果一个测试 payload 的空标签近邻与 CVE 近邻非常接近，说明该请求可能是“相似但不应输出 CVE”的流量，因此抑制输出。

关键参数：

```text
empty_penalty_margin=0.05
empty_penalty_floor=0.80
empty_penalty_ratio=0.50
```

这一步显著降低了初始版本的误报。

### OOF blocklist

`utils/learn_blocklist_from_folds.py` 使用训练集 out-of-fold 预测找系统性误报 CVE，而不是从测试集真值直接写规则。

当前使用的 blocklist：

```text
data/experiments/fold_blocklist_fp20_p002_mf2.txt
```

运行时通过参数启用：

```bash
--prediction-blocklist ./data/experiments/fold_blocklist_fp20_p002_mf2.txt
```

### Structured rerank

`src/structured_features.py` 抽取 HTTP 结构化特征：

```text
method
path
path token
query key
body key
payload token
```

`main.py` 支持可选 structured rerank：

```bash
--structured-rerank-alpha 0.03
--train-feature-path ./data/experiments/train_combined_cleaned.csv
```

重排公式：

```text
score = semantic_score + alpha * feature_bonus
```

`feature_bonus` 包含 method match、path exact match、path token Jaccard、query key Jaccard、body key Jaccard、payload token Jaccard。

该分支提高 recall 和 Macro-F1，但会牺牲一部分 precision，因此作为 recall-first 模式使用。

### 结构化规则配置

`utils/structured_signature_rules.py` 可以从训练数据中挖掘高精度结构化规则。规则类型包括：

```text
path
query_keys
body_keys
query_key_value
body_key_value
token
```

规则可以导出为可审计 JSON：

```bash
python utils/export_rule_config.py
```

配置字段包括：

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

应用规则：

```bash
python utils/apply_rule_config.py
```

当前推荐只对空预测补召回：

```bash
--only-empty
--max-additions 1
```

避免在已有预测上盲目追加标签导致 FP 扩散。

### 主流程集成

`main.py` 串联以下步骤：

```text
检查输入文件
清洗训练集
清洗测试集
构建或复用 FAISS 索引
编码或复用测试向量
FAISS top-k 检索
CVE 候选聚合
structured rerank（可选）
空标签近邻抑制
prediction blocklist（可选）
写出 id,cve_labels CSV
输出候选数量、相似度统计、预测数量分布
```

常用控制参数：

```bash
--overwrite-clean
--overwrite-test-clean
--overwrite-index
--reuse-cache
--top-k
--base-threshold
--prediction-blocklist
--structured-rerank-alpha
```

### 依赖

当前代码实际依赖精简为：

```text
numpy
pandas
tqdm
faiss-cpu
sentence-transformers
```

不再依赖 LangChain 或 Chroma。

## 启动项目

### 1. 构建 combined 训练集

```bash
python utils/merge_training_csv.py \
  --input ./data/train_with_ultimate.csv \
  --input ./data/train_payload.csv \
  --output ./data/experiments/train_combined.csv
```

### 2. 运行基础主流程

```bash
python main.py \
  --train-path ./data/train_with_ultimate.csv \
  --test-path ./data/test_payload.csv \
  --test-payload-column payload_decoded \
  --store-dir ./embeddings/faiss_store_combined \
  --output-path ./data/experiments/test_official_base.csv \
  --model-path ./models/all-MiniLM-L6-v2 \
  --reuse-cache
```

### 3. 启用 OOF blocklist

```bash
python main.py \
  --train-path ./data/train_with_ultimate.csv \
  --test-path ./data/test_payload.csv \
  --test-payload-column payload_decoded \
  --store-dir ./embeddings/faiss_store_combined \
  --output-path ./data/experiments/test_official_precision_first.csv \
  --model-path ./models/all-MiniLM-L6-v2 \
  --reuse-cache \
  --prediction-blocklist ./data/experiments/fold_blocklist_fp20_p002_mf2.txt
```

### 4. 启用 structured rerank

`--train-feature-path` 必须和 FAISS meta 的训练顺序一致。当前本地实验使用：

```text
data/experiments/train_combined_cleaned.csv
```

如果该文件不存在，可生成：

```bash
python - <<'PY'
import pandas as pd
paths = ["data/train_with_ultimate_cleaned.csv", "data/experiments/train_payload_cleaned.csv"]
pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
    "data/experiments/train_combined_cleaned.csv",
    index=False,
)
PY
```

运行：

```bash
python main.py \
  --train-path ./data/train_with_ultimate.csv \
  --test-path ./data/test_payload.csv \
  --test-payload-column payload_decoded \
  --store-dir ./embeddings/faiss_store_combined \
  --output-path ./data/experiments/test_official_rerank.csv \
  --model-path ./models/all-MiniLM-L6-v2 \
  --reuse-cache \
  --prediction-blocklist ./data/experiments/fold_blocklist_fp20_p002_mf2.txt \
  --structured-rerank-alpha 0.03 \
  --train-feature-path ./data/experiments/train_combined_cleaned.csv
```

### 5. 应用规则配置

导出规则配置：

```bash
python utils/export_rule_config.py \
  --rules ./data/experiments/structured_rules_train_s20_p1.csv \
  --output ./data/experiments/structured_rules_train_s20_p1_config.json \
  --source train-structured-s20-p1 \
  --min-support 20 \
  --min-precision 1.0 \
  --enabled \
  --risk-level low
```

应用规则：

```bash
python utils/apply_rule_config.py \
  --config ./data/experiments/structured_rules_train_s20_p1_config.json \
  --truth-or-test ./data/test_payload.csv \
  --pred ./data/experiments/test_official_rerank.csv \
  --output ./data/experiments/test_official_recall_first.csv \
  --only-empty \
  --max-additions 1
```

## 评估

整体指标：

```bash
python utils/evaluate_predictions.py \
  --truth ./data/test_payload.csv \
  --pred ./data/experiments/test_official_recall_first.csv
```

per-CVE 指标：

```bash
python utils/evaluate_per_label.py \
  --truth ./data/test_payload.csv \
  --pred ./data/experiments/test_official_recall_first.csv \
  --output-summary ./data/experiments/per_label_recall_first.csv
```

长尾诊断：

```bash
python utils/diagnose_long_tail.py \
  --truth ./data/test_payload.csv \
  --pred ./data/experiments/test_official_recall_first.csv \
  --train ./data/train_with_ultimate_cleaned.csv \
  --train ./data/experiments/train_payload_cleaned.csv \
  --output ./data/experiments/long_tail_diagnosis.csv
```

导出缺样本 CVE：

```bash
python utils/export_data_gap_report.py \
  --per-label ./data/experiments/per_label_recall_first.csv \
  --train ./data/train_with_ultimate_cleaned.csv \
  --train ./data/experiments/train_payload_cleaned.csv \
  --min-fn 20 \
  --max-train-support 0 \
  --output-csv ./data/experiments/cve_data_gap_no_train_support.csv \
  --output-md ./data/experiments/cve_data_gap_no_train_support.md
```

## 当前限制

仍有一批 CVE 在训练集中没有支持样本，例如：

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

这些不能靠可靠训练规则解决。继续提升上限需要补充授权样本或公开样本，然后重建索引和规则。

## 更多技术细节

完整优化过程、实验分支和技术分析见：

```text
EXPERIMENT.md
```
