# AGENTS.md

## 项目定位

个股崩盘预测的时序模型库（PyTorch）。设计事实来源为 `docs/RESEARCH_DESIGN.md`（改标签/特征/协议前必须先读）。当前为 LSTM 双标签周采样管线：gen1 单股冒烟、gen2 小池验证、gen3 扩展面板、gen4 数据扩展与新协议（662 只）、gen5 中证800 并集（1577 只）。

## 环境

- Python 3.12，虚拟环境固定在仓库根 `.venv/`
- 依赖见 `requirements.txt`；激活：`.venv\Scripts\Activate.ps1`

## 常用命令

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\run_weekly_baseline.py --ticker sz.300750
python scripts\run_weekly_baseline.py --ticker sz.000001 --pos-weight
python scripts\run_panel_baseline.py --only-fetch
python scripts\run_panel_baseline.py --pos-weight
python scripts\run_panel_baseline.py --pool-file _local/astock-pool-union.txt --data-dir data/dataset2 --gen gen4 --start 2013-01-01 --end 2026-10-10 --train-start 2017-01-01 --split-date 2025-01-01 --epochs 10 --batch-size 128
```

## 数据流

```
BaoStock / yfinance 下载（个股 + 指数）→ data/datasetN/<ticker>.csv（--data-dir 指定，--refresh 强制重下）
→ compute_features：21 维技术特征（仅用 t 及以前信息；beta60/rs20 需指数）
→ labels：标准A（未来5日 -8%）+ 标准B（Dimson 残差崩盘周，未来4周）
→ 周五采样，输入 (60 交易日 × 21 维)
→ 训练段拟合 winsorize + StandardScaler；时间切分 + 尾部 purge（默认 6 周）
→ LSTMBaseline × 双标签分别训练（BCEWithLogitsLoss，可选 --pos-weight）
→ 测试集 ROC-AUC / PR-AUC / F1 / Recall → outputs/models/<模型类型>/<代际>/
```

面板管线（`run_panel_baseline.py`，gen2–gen5）：

```
沪深300成分抽样 / --pool-file 清单（gen3 起：662 → 1577 只历史成分并集）→ 逐标的 compute_features / labels
→ 面板样本表（date/ticker/双标签）→ 中间缓存 panel_cache/（参数或股票池变化自动失效）
→ 训练段池化 winsorize + StandardScaler
→ 按日期切分 + 6 周 purge → 汇总 TensorDataset 训练（双标签）→ 整体与分年指标
→ outputs/models/<模型类型>/<代际>/（含 panel_samples.csv / panel_stats.json）
```

数据源选择：`--source auto`（默认）按代码格式判定——含 `.`（如 `sz.300750`）走 BaoStock（日线后复权），不含（如 `AAPL`）走 yfinance（auto_adjust）。

## 输出目录约定

- 产物一律放 `outputs/models/<模型类型>/<代际>/`，如 `outputs/models/lstm/gen1/`
- 模型类型 = 模型族名（lstm / tcn / patchtst…）；代际 = 设计迭代号（gen1、gen2…），**新代际不得覆盖旧代际**
- 单股（gen1）命名：`<模型>_<标签>_<标的>[_pw].pt`、`metrics_<标签>_<标的>[_pw].json`、`prep_<标的>.json`、`diagnostics_flags_<标的>.json`
- 面板（gen2–gen5）命名：`<模型>_<标签>_panel[_pw].pt`、`metrics_<标签>_panel[_pw].json`、`panel_samples.csv`、`panel_stats.json`、`prep_panel.json`、`panel_cache/`（中间缓存）
- `data/`、`outputs/` 不入库（见 `.gitignore`）

## 数据目录约定

- `data/dataset1/`：40 只小池（gen2 时代）；`data/dataset2/`：662 只沪深300 历史并集（gen3–gen4）；`data/dataset3/`：1577 只中证800 历史并集（gen5）
- 每份快照含全部个股 CSV + 指数 `sh.000300.csv`；脚本用 `--data-dir data/datasetN` 指定
- 数据根目录已清理为快照结构；抓取/更新工具在 `_local/`（不入库），新数据写入对应快照目录

## 约定与约束

- 项目内引用一律使用相对路径，禁止绝对路径入库
- 文档中对外部材料统一称"前期研究"（课题前期四份调研材料），不写具体文件名
- 标签窗口、特征集、切分方式任何改动：同步更新 `README.md`、`docs/RESEARCH_DESIGN.md` 与本数据流
- 评估禁止使用 Accuracy，理由见 README
- yfinance 被 Yahoo 限流时（403/429）改用 BaoStock 源，不要频繁重试以免加重封锁
- `_local/` 存放本地素材（股票清单等），不入库
- 处理 "改配置/脚本 + 重跑" 类操作前先向用户说明并获批准

## 已知坑

- BaoStock 行遍历必须成对调用 `rs.next()` 与 `rs.get_row_data()`；只调 `next()` 不调 `get_row_data()` 会因游标不前进而死循环
- BaoStock 服务端偶发降级（登录/查询正常但拉取停滞）；`data.py` 已加迭代上限与 3 次重试，仍失败时等待数分钟再试
- 指数（如 sh.000300）未必支持股票特有字段（如 turn）；`data.py` 先试全字段、失败自动回退基础字段
- 沪深300成分接口行格式为 `[date, code, name]`；`data.py` 按 fields 定位 code 列，勿按位置硬取
- 全池抓取受 BaoStock 服务端速度影响可能需 30–60 分钟且偶发单只失败；支持断点续抓，重复运行 `--only-fetch` 直至补全
- 面板中间缓存 `panel_cache/`（gen3 约 250 MB，gen4/5 更大）：beta/stat/sigma/seq-len/日期/股票池任一变化自动触发重建；强制重建用 `--rebuild-cache`
- 测试样本末端约 6 周「前瞻损失」（B 标签 4 周 horizon + Dimson 2 周干净期），属设计内行为，勿当 bug 处理
- 换日期范围必须加 `--refresh`，否则命中旧缓存导致结果不变

## 路线

见 `PROGRESS.md`（唯一路线来源，勿在本文件复制）。
