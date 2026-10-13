# PROGRESS

仅记录未完成事项与路线，不放会过期的"当前状态"。

## 路线

- [x] 接入 GRU / TCN 基线并横向对比（gen5：LSTM/GRU/TCN/LightGBM/LogReg，普通+加权）
- [x] 图表产物：`scripts/make_figures.py` → `docs/figures/`（结果图 + 结构图，PNG+SVG）
- [ ] 验证协议：Purged & Embargoed walk-forward 滚动验证 + 多种子（当前结论基于单窗口单种子）
- [ ] 阈值与校准分析深化：PR 曲线、最优阈值搜索、概率校准（基础版图已出）
- [ ] 逻辑回归复跑：提高 `max_iter` / 换求解器，修正未收敛问题
- [ ] 不平衡处理深入：Focal Loss、采样策略（`--pos-weight` 对照已完成）
- [ ] 标签体系扩展：NCSKEW（负收益偏度）/ DUVOL（上下波动率比）并行头
- [ ] 全市场面板与行业分层统计（gen5 之后的进一步扩展；ST/退市系统化规则待定）
- [ ] 数据源扩展：Tushare 接入（需注册 Token）
- [ ] 少样本 / 领域自适应支线（对接前期研究第二阶段方向）
