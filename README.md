# 共同需求波动下的流动教学运力与服务承诺

这是一个面向社区老年教育运营的本地分析原型，使用合成记录演示从业务记录到运营摘要、再到联合运力决策的处理。项目把记录读取、需求预测、情景证据和运营决策分开，便于核对原始事实与派生结果。

## 环境与使用

需要 Python 3.11 或更高版本，无第三方依赖。运行 `python3 -m unittest discover -s tests -v` 执行测试；运行 `python3 -m learning_mobile_capacity.report` 查看旧版单点汇总，运行 `python3 -m learning_mobile_capacity.scenario_report` 查看联合需求情景与运力决策报告。所有记录均为本地合成数据，不包含个人信息。
也可使用 Docker 运行：`docker build -t learning-mobile-capacity .`，再执行 `docker run --rm learning-mobile-capacity`。容器内运行测试可执行 `docker run --rm --entrypoint python3 learning-mobile-capacity -m unittest discover -s tests -v`。

## 联合决策口径

- **信息截点**：`forecast.build_history` 只接收 `as_of` 之前已到达的记录（`received_at` 字段，缺省按周期首日），日期统一规范为年-月-日元组比较，开班后的实际到场无法回填进当时预测。
- **共同起伏与相依性**：共同观察期上估计公共因子（节假日、同一活动的同步涨跌）；点位对有足够重叠期才给出收缩后的经验相关性，重叠不足时显式标记“未知”，不假设独立、不伪造系数。
- **联合情景**：输出证据情景、独立假设情景、同单调上界与反单调下界（Fréchet–Hoeffding 包络），每个情景写明依据与尚未澄清的依赖。
- **运力与承诺**：`capacity.decide` 在最低服务次数（硬约束，不得静默取消）之上枚举班次组合，按当天工时、路程与随班资格核算最少工作日，用任意相依结构下都成立的违约概率包络判定可靠性；风险按“需求本身 / 资料不足 / 路线资源冲突”归因，权衡只保留帕累托前沿。
- **不可承诺的表达**：最低承诺冲突或所有可行安排都达不到约定可靠度时，状态显式标记为不可承诺，附无法作出的承诺与可调整范围，不输出带成功标记的空计划。

## 现有行为

旧版摘要只反映样例记录的基础汇总，不用于形成个体资格或运营承诺。SQLite 数据库在调用时创建于临时目录，原始 CSV 保持不变。`data/` 下新增 `demand_history.csv`（含截点之后到达的记录用于验证过滤）、`resources.json`（工时、路程、教师、设备）与 `commitments.json`（截点、可用工作日、可靠度目标与最低班次）。
