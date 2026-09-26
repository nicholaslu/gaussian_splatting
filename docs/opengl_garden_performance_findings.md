# Garden 全量场景下的 OpenGL 动态视角性能观察

## 背景与测试范围

本记录汇总当前 GaussianSplatting RViz 插件在 Garden 全量数据下的性能现象，并对 Metal 与 OpenGL 4.3 GPU preparation 路径进行初步对照。

本轮重点不是静态帧率，而是相机移动时的持续性能和交互延迟。测试场景约有 **5,834,784 个 splat**；常见视角下约有 **90 万至 100 万个 splat 可见**。远端 OpenGL 测试设备使用 NVIDIA RTX 4090。

以下内容分为实测现象、已经基本排除的方向和根据代码得到的初步结论。尚未通过 GPU 分阶段计时验证的内容均标记为推断。

## 已观察到的现象

### 静止与移动差异明显

- 相机静止后，RViz 可以恢复到约 60 FPS。
- 相机持续移动时，帧率明显下降，并伴随交互响应变慢。
- 停止发布器后，已经加载的场景在移动视角时帧率没有明显恢复。

这说明瓶颈不是 ROS 消息持续发布、反序列化或重复上传，而是已加载场景在视角变化时触发的逐视角 GPU 工作。

### OpenGL GPU preparation 已经启用

- OpenGL 4.3 GPU preparation 路径正常启用。
- CPU 仅需约 **0.06–0.075 ms** 将一次 preparation 提交到 GPU。
- RViz 进程 CPU 使用率约为 **22%**，发布器在稳定状态下基本空闲。
- 移动视角时 NVIDIA GPU 利用率持续达到 **96–100%**。

因此，当前低帧率不是原来的 CPU 排序问题重新出现，而是 GPU preparation 或其调度方式成为了主要限制。

### 降低输出分辨率没有消除饱和

将 Offscreen Render Scale 从 `1.0` 降至 `0.25` 后，像素数量理论上减少到原来的 1/16，但移动时 GPU 仍接近 98–100% 利用率。

这基本排除了“最终 splat 光栅化和片元填充是当前唯一主因”。填充和混合仍有成本，但无法解释分辨率大幅下降后 GPU 仍然饱和的现象。

### 减少可见 splat 后仍然饱和

将 Minimum Screen Radius 设为 `8` 后，可见数量从约 100 万下降到 **515,174**，但移动时 GPU 仍然接近满载。

该参数只会减少裁剪后的 survivor 数量，不会避免第一阶段扫描全部 5,834,784 个输入。因此这一结果指向裁剪、压缩以及围绕全量输入执行的准备阶段，而不只是 survivor 的最终绘制。

### GPU 利用率高，但功耗和显存利用率不高

测试时观察到 GPU utilization 接近 100%，但功耗约为 95 W，显存利用率约为 0–2%。这些指标不能直接等价为某个具体 kernel 的耗时，不过它们更像是低占用率、同步/调度受限或串行指令较多的工作负载，而不像纯显存带宽或片元吞吐饱和。

## 当前 GPU preparation 流程

Metal 与 OpenGL 的高层算法基本一致：

1. 对全部 splat 做裁剪并生成深度键。
2. 统计和扫描 survivor 数量。
3. 将 survivor 压缩到连续的 key/index 缓冲区。
4. 执行四轮 8-bit stable LSD radix sort。
5. 根据排序结果做投影、SH 求值和 gather。
6. 写入 indirect draw 参数并绘制。

因此，OpenGL 变慢并不是因为它仍然走完整的 CPU 排序路径，也不是因为 Metal 使用了完全不同的高层算法。

## Metal 与 OpenGL 的关键执行差异

### SIMD lane 排名

Metal 的 compact scatter 和 radix scatter 使用原生 `simd_ballot` 计算 lane rank。

当前 OpenGL 4.3 GLSL 路径没有使用 subgroup ballot，而是在一个 32-lane tile 内通过循环比较其他 lane 来计算稳定排名。这会产生更多指令和串行依赖，在 NVIDIA GPU 上很可能比原生 warp/subgroup 操作低效。

### Dispatch 和同步

Metal 将 preparation 的多个阶段编码到同一条 Metal command timeline 中，并使用较细粒度的 buffer barrier。

OpenGL 路径需要多次：

- 切换 compute program；
- 重新绑定 SSBO；
- 发起 compute dispatch；
- 执行 `glMemoryBarrier`。

一次 preparation 包含十几次 dispatch 及其阶段同步。即使每个 kernel 本身不长，这些全局 barrier、驱动调度和 kernel 间空隙也可能形成明显开销。

### 视角更新的排队策略

目前 Metal 和 OpenGL 都没有明确的“上一帧 preparation 尚未完成时，只保留最新视角”的合并或节流机制。

相机拖动会连续产生视角更新；CPU 提交一次 OpenGL preparation 只需约 0.07 ms，因此 CPU 可以远快于 GPU 地继续提交完整流水线。若中间视角的工作不能及时丢弃，就会形成 GPU 队列积压，表现为：

- 移动时低帧率；
- 输入到画面的延迟逐渐增加；
- 停止移动后需要等待队列消化。

这一项是两个后端共有的结构性风险，并非 OpenGL 独有。

## 初步结论

当前证据支持以下判断：

1. **主要瓶颈位于相机变化时触发的 GPU preparation，而不是 publisher 或 CPU 排序。**
2. **当前问题不是单纯的片元填充瓶颈。** Render Scale 降至 0.25 后仍然饱和说明 preparation 的权重很高。
3. **全量扫描是场景规模扩展的基础成本。** 即使只有约 51 万 survivor，第一阶段仍要处理全部约 583 万输入。
4. **OpenGL 的实现效率很可能低于 Metal。** 最可疑的差异是手动模拟 32-lane stable rank，以及大量 dispatch/global barrier。
5. **缺少视角更新合并可能放大了所有 GPU 成本。** 单帧 preparation 即便只是略慢于目标帧时间，连续拖动仍可能提交远多于 GPU 能消费的工作。
6. **Metal 并没有从算法上避开全量扫描。** 它可能因为原生 SIMD primitive 和 command-buffer 模型表现更好，但在同样的 583 万规模与连续视角更新下也存在退化可能。

目前还不能仅凭总体 GPU utilization 判断 compact、radix、gather 或等待 barrier 中哪一个占比最高。对此需要加入真正的 GPU 分阶段计时。

## 建议的下一步验证顺序

### 1. 加入 OpenGL GPU timestamp query

分别记录以下阶段的 GPU 时间：

- cull/key generation；
- compact count/scan/scatter；
- 每轮 radix histogram/scan/scatter；
- projection/SH/gather；
- 最终 draw。

同时记录从第一阶段开始到最终 draw 完成的总时间，避免只看到各 kernel 而漏掉 barrier 和 dispatch 间空隙。

### 2. 加入 latest-view 合并和 in-flight 限制

GPU 正在处理视角 A 时，如果又收到 B、C、D，应允许丢弃未提交的中间视角，只在可再次提交时处理最新的 D。静止后仍应保证最终视角被精确处理。

这是最直接的交互改善手段，也能区分“单帧 preparation 太慢”和“重复提交造成队列积压”各自的影响。

### 3. 为 OpenGL 增加原生 subgroup 路径

在运行时检测 NVIDIA/GLSL 可用的 subgroup ballot 能力，用原生 warp/subgroup rank 替换 GLSL 中的 32-lane 手动循环，并保留当前纯 OpenGL 4.3 实现作为兼容回退。

重点比较 compact scatter 和 radix scatter 的分阶段时间，而不是只比较最终 FPS。

### 4. 再评估分层裁剪和 CUDA/CUB

如果 cull/key generation 的全量扫描仍占主导，应考虑层次包围结构或分块可见性，避免每个视角扫描全部 splat。

CUDA/CUB 可以为 NVIDIA 提供成熟的 radix primitive，但它主要改善排序阶段，不能单独解决全量裁剪和重复视角提交问题。应在 timestamp 结果确认排序占比后再决定优先级。

## 与独立问题的边界

超过 5,592,405 个 splat 时的 OpenGL 浮点纹理索引精度问题属于 CPU-prepared GLSL 1.20 fallback。当前 OpenGL 4.3 GPU preparation 使用整数 SSBO 索引，不受该问题影响；它不是本次动态视角低帧率的原因。
