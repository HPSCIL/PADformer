# Input data specification

本目录用于用户准备的真实数据或生成的合成示例。真实数据、坐标和地形矩阵不随源码分发。

## air_quality.csv

必须有以下列，四种污染物的读取顺序固定：

```csv
PM2.5,PM10,O3,NO2,u10(m/s),v10(m/s)
```

污染物单位应与目标实验统一；风速分量以 m/s 表示。不能包含未处理的缺失值或无穷值。CSV 总行数必须能被站点数整除。时间戳可以作为额外列保存，但读取器不会用时间戳排序或检查时间间隔，必须预先对齐、补齐时间轴并排序。

- `station_block`：站点 0 的全部时间步，随后站点 1 的全部时间步，依此类推。
- `time_block`：时间 0 的全部站点，随后时间 1 的全部站点，依此类推。

所有站点应包含完全一致的时间轴，且污染物、风场、坐标和矩阵使用相同站点顺序。原始数组形状为 `[time, station, 4]` 和 `[time, station, 2]`。

## stations.xlsx

每行一个站点，使用 `latitude, longitude`（纬度在前）作为列名；代码也接受 `lat, lon` 或 `纬度, 经度`。应恰好提供 N 个有效站点，坐标使用十进制度。

## terrain.npz

必须提供 `relief`、`climb`、`slope`、`tpi`、`svf` 五个数组，每个形状为 `[N,N]`，索引顺序与站点表一致。也支持 `arr_0` 至 `arr_4`。推荐使用明确键名。

```python
import numpy as np
# relief, climb, slope, tpi, svf 为用户依据研究方法计算的 N×N 数组
np.savez_compressed('data/terrain.npz', relief=relief, climb=climb,
                    slope=slope, tpi=tpi, svf=svf)
```

默认取 slope、tpi、svf 三通道；`--terrain_channels 5` 使用全部五通道。加载器对 SVF 取负、逐通道标准化并将对角线置零，请提供尚未由该加载器处理的矩阵。项目不包含从 DEM 自动计算这些矩阵的算法，需依据论文方法自行准备。

## Minimum data length

训练先按时间步进行 60/20/20 划分，每段至少需要 `lag + pre_len` 个时间步。默认 72/72 窗口需要约 720 个总时间步；示例生成器默认提供 730 步。独立预测仅需要每个站点至少 `lag` 个历史时间步。

训练入口在地形或站点文件缺失时有零偏置回退行为；完整物理模型实验应提供全部输入并检查日志中的加载结果。独立预测入口要求显式提供这些文件。
