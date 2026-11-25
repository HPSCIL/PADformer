# Repository content mapping

参考来源：https://github.com/HPSCIL/MSTVFFN-multiple-air-pollutants-prediction/tree/main/MSTVFFN

核对日期：2026-11-25。通过 GitHub raw 文件读取参考仓库的 README 和十个 MODEL Python 文件。以下对应表示功能组织，不表示两个模型采用相同算法或具有相同参数。

| MSTVFFN 内容 | PADformer 内容 | 说明 |
| --- | --- | --- |
| 根目录 Readme.md | README.md | 简介、框架、依赖、数据、目录、参数、运行和引用 |
| Fig/MSTVFFN.png | Fig/PADformer.svg | 根据 PADformer 实际模块绘制 |
| MODEL/main.py | main.py | 训练、验证、最佳权重测试；本项目另支持多种子和消融 |
| MODEL/model.py | model.py | PADformer 类、双分支和融合 |
| MODEL/data_preprocess.py | data_preprocess.py | 滑动窗口、Dataset；本项目同步处理风场 |
| MODEL/utils.py | utils.py | 日志、指标、随机性设置 |
| MODEL/Embedding.py | Embedding.py | 本项目为 FFT 与趋势嵌入 |
| MODEL/Layers.py | Physics_block.py、model.py | 编码和融合由现有模块承担 |
| MODEL/Temporal_block.py | Physics_block.py、Decoder.py | 时间注意力 |
| MODEL/Spatial_block.py | Physics_block.py | 本项目为物理偏置空间注意力，不采用参考模型的 Chebyshev 图卷积 |
| MODEL/Spectral_block.py | Embedding.py | 本项目采用 FFT 嵌入，算法不同 |
| MODEL/CrossAttention.py | Physics_block.py、Decoder.py | 交叉注意力置于编码器和解码器内部 |
| Data/Beijing.xlsx | data/README.md、examples/generate_demo_data.py | 数据约定和可生成的合成示例；不转用北京样例为本项目实测数据 |
| adj/adj_bj.xlsx | 用户提供的 terrain.npz、stations.xlsx | 输入角色不同，本项目不需要 Chebyshev 邻接矩阵 |
| 训练日志、权重 | results/（运行生成） | 不把对方权重或日志作为 PADformer 结果 |
| 无单独预测入口 | predict.py | 新增最新历史窗口预测 |

保留源码位于仓库根目录的现有组织方式，避免复制一套 MODEL 文件导致入口和引用不一致。借用模块的来源声明不因模型名称统一而消失；具体代码出处仍需要作者提供或核实，不能仅凭类名推断。
