# 硕士学位论文开题报告

## 面向统一理解与生成多模态大模型的可配置训练后量化研究

> 说明：本文按学术型硕士研究生开题要求撰写，研究周期一年。第一部分文献综述约 5,100 字，全文约 9,400 字。

## 1. 课题国内外研究进展、现状、挑战与意义

### 1.1 研究背景

基于 Transformer [1] 的语言模型在过去几年里快速发展。GPT-3 [2] 展示了大模型在少样本条件下的通用能力，LLaMA [3] 等开放权重模型则降低了研究和部署的门槛。与此同时，视觉与语言的联合建模也从 CLIP [4]、SigLIP [5] 这类对比学习模型，发展到 Flamingo [6]、BLIP-2 [7]、LLaVA [8] 和 InternVL [9] 等具备图文理解能力的多模态大模型。这类模型已经用于图像描述、视觉问答、文档理解和智能体等场景，成为很多应用的基础组件。

2024 年以来，研究重点开始从“只做理解”转向“理解和生成共用一个模型”。Janus [10] 将理解与生成的视觉编码解耦：理解分支使用 SigLIP 的连续特征，生成分支使用 VQ 图像 token，两者共享同一个自回归语言骨干。Janus-Pro [11] 通过扩大数据和模型规模提升了统一模型的生成质量，JanusFlow [12] 则把自回归生成与整流流结合起来。其他路线包括 Emu [13] 的“语言模型加扩散解码器”、Emu3 [14] 的纯下一 token 预测、Chameleon [15] 的早融合混合模态建模，以及 Show-o [16]、Show-o2 [17] 在离散 token 和连续 latent 上的尝试。统一模型因此成为多模态研究的一条重要分支。

统一模型的能力提升也带来了明显的部署压力。以 Janus-Pro-7B 为例，语言骨干约 70 亿参数，fp16 权重约 14 GB，加上视觉编码器、VQ 解码器、KV Cache 和生成过程中的中间激活，单卡显存需求更高。在边缘设备、单卡工作站和高并发服务中，显存容量和带宽往往比峰值算力更容易成为瓶颈。量化通过低比特整数近似权重和激活，可以显著降低存储和带宽开销，因此成为最常用的压缩手段之一。问题在于，低比特误差在生成任务中会沿着 token 序列累积，模型可能表面上还能出图，但属性、数量和空间关系已经明显失真。如何让统一模型在低比特下同时保住理解和生成，是本文要回答的问题。

从压缩手段看，量化、剪枝、蒸馏和低秩分解都能降低部署成本，但量化有几个现实优势。它一般不需要重新训练，可以直接作用在已有权重上；实现和硬件支持相对成熟，INT8 和 INT4 在主流 GPU 上都有对应 kernel；收益也比较直接，权重存储和显存带宽会随位宽下降。代价是精度损失不容易控制：校准集选得不好、粒度太粗或者离群值处理不当，都可能在某个任务上突然掉点。统一模型的风险更高，因为理解分数可能还正常，生成质量已经坏了。如果只看一个榜单，很容易得出错误结论。所以本文把理解和生成同时评测作为基本要求，而不是先在理解上调好再去补生成。

### 1.2 统一理解与生成模型

现有统一模型大致可以分为四类。第一类以 Janus 和 Janus-Pro 为代表，理解和生成使用不同的视觉编码，但共享同一个自回归语言骨干。第二类以 Chameleon 为代表，文本和图像 token 在同一个序列里混合训练，属于早融合路线。第三类以 Emu3 为代表，把图像、文本和视频都离散化为 token，统一做下一 token 预测。第四类以 Emu 和 Show-o2 为代表，语言模型负责文本，扩散或流匹配头负责图像生成。四类结构对量化的影响并不相同：共享骨干会让理解和生成的误差相互耦合，早融合会让图像 token 的扰动通过注意力扩散到全局，长 token 序列会放大累积误差，扩散头则涉及随 timestep 变化的激活分布。

Janus-Pro 的结构适合做量化研究。它的理解分支是单次前向、输出较短，生成分支则要把 384×384 图像编码成 576 个离散 token，再由语言骨干逐步解码。理解侧没有明显的误差累积，生成侧有清晰的 576 步轨迹，两者刚好形成对照。如果同一组权重在理解上几乎不掉点、在生成上却明显退化，就可以把问题定位到生成路径和误差累积上，而不是笼统地归因于“模型对量化敏感”。

从评测看，统一模型已经有一套相对成熟的基准。GenEval [36] 关注物体数量、颜色、位置和属性绑定，DPG-Bench [37] 使用密集 prompt 和多模态模型评分，T2I-CompBench [38] 从空间关系、数量和属性等维度评价组合能力。理解侧则有 MMMU [39]、MME [40]、TextVQA [41]、MathVista [42]、POPE [43] 和 SEED-Bench [49] 等。评测工具变多了，但结论不一定更可靠：生成采样本身有随机性，小样本子集的波动可以达到几个百分点，如果不固定解码协议、不做配对统计，很容易把噪声当成方法收益。这一点在后面还会展开。

统一模型的效果和训练数据关系很大。Janus-Pro 的经验是把理解数据和生成数据按阶段配比，理解侧需要图文对、指令和多轮对话，生成侧需要覆盖物体、属性、空间关系和文字渲染的细粒度描述。数据配比不合适时，模型会在组合性 prompt 上出现属性绑定错误，或者理解侧的语义表征变差。这个特点会直接影响量化校准：如果校准集只取理解数据，生成分支的 scale 会被低估；如果只取生成 prompt，理解侧的文本路径又可能被过度代表。另一个现实问题是模型规模，7B 模型能力强但评测成本高，1B 模型便宜但量化误差的相对影响可能更大。本文以 7B 为主、1B 只做趋势验证，原因就在这里。

### 1.3 语言模型量化

语言模型量化的基本做法是把浮点权重映射到低比特整数网格，再用缩放因子和零点还原。RTN [18] 直接舍入到最近邻，实现简单，但大模型权重和激活中存在少量幅值很大的离群值，直接舍入会带来明显误差。LLM.int8() [19] 把离群维度保留为 8-bit，其余维度做 int8 计算，说明了离群通道对量化误差的影响。ZeroQuant [20] 从系统和算法协同的角度做了逐层量化与知识蒸馏。GPTQ [21] 把逐层量化写成带 Hessian 近似的二次误差最小化问题，按列量化并更新剩余权重，使 3–4-bit 权重量化在语言模型上达到可用水平。AWQ [22] 根据激活尺度对权重做等价缩放，保护少量显著通道。SmoothQuant [23] 把激活的量化难度迁移到权重侧，使 8-bit 激活变得可行。OmniQuant [24] 进一步引入可学习的平滑和裁剪参数。QuaRot [25] 用旋转矩阵打散离群值，SpQR [26] 用稀疏表示保留离群值，AQLM [27] 则探索 1–2-bit 的加性向量量化。这些方法构成了当前语言模型 PTQ 的主要工具箱。

从这些工作里可以总结出几点经验。逐层、逐通道的误差补偿通常比全局统一舍入更有效；粒度和硬件效率需要折中，group size 128 是常见选择；校准集要覆盖目标任务的激活分布；激活量化比权重量化困难，W4A16 已经比较成熟，W4A8 在部分模型上可用，W4A4 和 W3A8 仍不稳定；离群值处理是低比特量化的核心矛盾。现有的 GPTQ、AWQ、SmoothQuant 基本都建立在纯文本、单次前向的假设上，而统一模型多了图像 token、跨模态注意力和数百步自回归解码，同一层在不同解码步的激活分布会变化，单纯套用这些方法并不够。

具体到实现，粒度和校准是两个绕不开的选择。per-tensor 只用一个 scale，元数据少、kernel 简单，但不同通道的动态范围被强行统一，小幅值通道容易被压扁；per-channel 和 per-group 能缓解这个问题，但会增加元数据，对 kernel 也有要求。校准方面，样本太少会被离群值主导，样本太多收益很快饱和；校准时的序列长度和 batch 组织也会影响激活统计，短序列校准未必能代表长序列生成的真实分布。语言模型量化里这些问题已经有较多讨论，但统一模型的模态更多、解码更长，原来的经验不能直接照搬。

### 1.4 多模态模型量化

多模态模型的量化研究起步稍晚。早期工作通常把视觉编码器留在 fp16，只量化语言骨干，因为视觉特征的分布和语言模型差异较大，直接量化容易损伤视觉表征。Q-VLM [28] 是比较有代表性的系统工作，它面向大型视觉语言模型提出 W4A4 的 PTQ 方案，分析视觉编码器和语言解码器之间的误差传播，并加入视觉编码误差补偿，使 4-bit 权重和 4-bit 激活在多个视觉问答基准上接近 fp16。这项工作说明多模态量化不能照搬纯文本经验，但评测仍集中在理解任务，没有涉及图像生成。

MBQ [29] 是与本课题最接近的工作。它指出视觉 token 和文本 token 对量化的敏感度不同，如果使用统一的 scale 搜索目标，视觉路径的误差会被文本路径的梯度主导。MBQ 分别为视觉和文本模态收集梯度，用中位数比值构造模态级重加权系数，并结合 block-wise 的 scale 搜索，在多个视觉语言模型上提升了低比特理解精度。它的贡献在于把模态均衡引入量化校准，但重加权粒度主要停留在模态级或标量级，没有深入逐通道差异；评测也集中在理解任务，没有覆盖统一模型的生成分支。国内团队在 Janus、Emu、InternVL 等模型上已经形成较完整的开源生态，Q-VLM 和 MBQ 也代表了视觉语言模型量化的前沿，但面向统一理解与生成模型的量化工作仍然很少。

量化对象也需要区分。视觉编码器参数不多，但对特征方向比较敏感，粒度太粗会破坏跨模态对齐；语言骨干占参数主体，是压缩收益的主要来源，但激活里有少量幅值很大的离群通道；VQ 解码器直接决定生成图像的质量，通常不适合低比特；KV Cache 量化则影响长序列生成的显存占用。现有工作大多只量化语言骨干，把其他部分留在 fp16，这在理解任务上通常安全，但生成任务是否够用还没有系统答案。本文会把量化对象作为配置空间的一维，至少把“哪些模块必须保留高精度”这个问题回答清楚。
### 1.5 生成模型量化

图像生成模型的量化研究主要集中在扩散模型。DDPM [33] 建立了去噪扩散的概率框架，LDM [32] 把扩散过程放到潜空间，DiT [31] 用 Transformer 替代 U-Net，成为后续大规模生成模型的主流结构。Q-Diffusion [30] 针对扩散模型做逐层、逐通道的权重与激活量化，并提出时间步感知的校准策略，使 8-bit 扩散模型在 FID 上接近全精度。后续工作进一步发现，扩散 Transformer 中对量化比较敏感的是注意力的输出投影和 MLP 的下投影，这些层的激活离群值会放大量化误差；不同 timestep 的激活分布差异也要求校准集覆盖整个去噪过程。这些结论和语言模型量化里的离群通道问题是相通的。

另一条路线是离散图像 token 的自回归生成。VQ-VAE [34] 和 VQGAN [35] 把图像编码成 codebook 索引，使图像生成可以复用语言模型的下一 token 预测框架。Janus-Pro 就属于这条路线：生成分支把图像编码成 576 个 token，再由共享语言骨干逐步解码。和扩散模型不同，自回归生成的第 t 步输出会直接成为第 t+1 步的输入，量化误差没有重新加噪的缓冲，容易沿着 token 序列形成正反馈。576 步累积下来，即使每一步的分布偏移不大，也可能表现为颜色漂移、物体结构错乱或语义丢失。

扩散量化和自回归量化的差异对方法设计有直接影响。扩散模型的校准通常按 timestep 分层，自回归模型则应按解码进度分层，否则 scale 只反映早期 token 的分布。扩散模型可以在每个 timestep 做局部校正，自回归模型更适合在关键解码步做干预，或者在层级别保护生成敏感层。现有扩散量化方法假设每步输入由外部噪声和时间步决定，无法直接刻画自回归 token 的误差反馈，因此统一模型的生成侧量化既是空白，也是机制研究价值比较高的部分。

生成侧还有两个细节值得注意。一个是无分类器引导（CFG），量化模型对 CFG 尺度的响应可能和 fp16 不一样，同样的引导强度在低比特下可能放大误差；另一个是 VQ 解码器，离散 token 的少量错误经过解码器后可能被放大成明显的纹理或结构问题。扩散模型量化通常会单独分析 timestep 和引导尺度，自回归模型则要分析解码进度和 token 位置。本文会把解码步敏感度和层敏感度放在一起看，而不是只比较最终分数。

### 1.6 评测体系与可复现性

统一模型的评测分为理解和生成两侧。理解侧常用 MMMU [39] 评测跨学科推理，MME [40] 评测感知和认知能力，TextVQA [41] 评测场景文字理解，MathVista [42] 评测视觉数学推理，POPE [43] 评测物体幻觉，SEED-Bench [49] 评测生成式理解；COCO [44] 提供图像描述和视觉问答的基础数据。生成侧常用 GenEval [36] 评测组合性，DPG-Bench [37] 评测复杂场景，T2I-CompBench [38] 评测空间关系和属性绑定；感知质量方面则使用 FID、CLIPScore [45]、PickScore [46]、HPSv2 [47] 和 ImageReward [48]。这些指标从不同角度衡量生成质量，但侧重点并不相同。

指标之间的差异会带来判断偏差。CLIPScore 对整体语义敏感，对数量、颜色绑定这类细粒度错误不够敏感；PickScore、HPSv2 和 ImageReward 反映人类偏好，容易受美观度和风格影响；FID 衡量分布距离，和人类偏好并不总是一致。T2I-CompBench 使用检测器评价空间关系和数量，区分度较高，但依赖检测器的可靠性；DPG-Bench 用多模态模型评分，覆盖复杂 prompt，但评分模型本身也可能被“看起来合理”的错误图像欺骗。因此，量化研究最好同时报告语义、组合性和偏好三类指标，并对关键结论做人工抽查。

统计功效也需要注意。生成任务的成功率服从二项分布，n=600 时标准误大约 2 个百分点，95% 置信区间约 ±4 个百分点；n=100 时区间会扩大到 ±10 个百分点左右。如果只在 100–600 张图上比较两个方法，2–3 个百分点的差异很可能只是噪声。本文会采用全量主榜加子集粗筛的方式：子集用于快速排除明显无效的配置，全量用于确认小效应；所有关键比较都报告配对 bootstrap 区间和多种子方差，并预先设定“全量差异小于 2 个百分点、子集差异小于 4 个百分点不解读”的规则。

评测报告也需要更透明。除了平均分，最好给出逐 tag 结果、样本量和置信区间；对失败案例做少量人工检查，避免评分模型把结构错误的图判成正确；校准集、scale 版本、解码参数和评测脚本都应该记录在案。生成任务的复现成本不低，如果这些信息缺失，后来者很难判断一个方法是真实有效还是评测口径不同。本文的工具包会把结果清单和评测配置一起保存，目的就是让结论可追溯。

### 1.7 现状、挑战与意义

从现有文献看，语言模型和理解型多模态模型的低比特量化已经比较成熟，W8/W4 基本可用，W3 在理解任务上也往往能撑住。扩散模型的量化也有 Q-Diffusion 这样的系统工作，8-bit 甚至更低比特在生成质量上已经接近全精度。真正缺的是统一理解与生成模型的量化：现有 MLLM 量化默认模型只做理解，扩散量化默认模型只做生成，而统一模型要求同一套权重同时服务两类任务，生成分支还存在长序列自回归的误差累积。这个交叉空白是本文的切入点。

具体困难主要有几个。不同模态的梯度尺度和激活分布差异很大，统一校准会让某一模态主导 scale，另一模态被系统性低估。生成侧的错误会沿解码步累积，理解侧的误差则没有这种反馈，所以“理解掉点少”不能说明“生成也没问题”。逐层混合精度可以缓解均匀量化的不足，但选哪些层、选多少层是一个带预算约束的组合优化问题，敏感度分数本身也有校准噪声。生成评测的方差大，小样本结论不可靠，必须配合确定性解码和统计检验。再往后还有激活量化不稳定、伪量化与真实部署之间存在差距、跨模型泛化不确定等问题。对硕士课题来说，前四个问题是主线，后三个更适合作为限制和后续工作来讨论。

这项工作的意义可以从三个层面看。科学上，它把统一模型在低比特下“哪里会坏、为什么坏、怎么保护”讲清楚，尤其是理解与生成的不对称机制，对多模态量化研究有补充价值。方法上，它把模态重加权、逐通道粒度、混合精度、校准配比这些零散开关放到同一个配置空间里比较，并用有效性地图区分真实收益和零结果，避免只报告正向模块。应用上，它面向单卡和边缘部署，给出从 W4 到 W3.4 的精度与显存权衡，并通过开源工具包和确定性评测降低复现成本。和已有工作相比，本文不打算提出新的量化算子，也不打算宣称在所有榜单上全面更优，重点是把统一模型这条新对象上的量化规律做扎实。

从研究格局看，国际上 DeepSeek、Google、Meta、OpenAI、BAAI 和新加坡国立大学等团队在统一模型和量化上都有积累，国内 DeepSeek、BAAI、上海人工智能实验室以及清华等团队在模型、评测和量化方法上也形成了比较完整的链条。差异主要不在单点方法，而在评测口径和工程可复现性。本文不追求提出一个全新的量化算子，而是希望在统一模型这个新对象上，把配置、机制和评测三件事做完整。

需要说明的是，本文讨论的低比特主要指权重 4-bit 和 3-bit，激活默认保持 16-bit；A8 作为单独问题讨论，不作为主方法的一部分。这样可以先把权重压缩这条主线做清楚，避免把激活量化的不稳定因素混进来。评测上以 GenEval 和 DPG-Bench 为主，理解侧用 MMMU 和 MME 做辅助，T2I-CompBench 和 COCO 作为补充，既能看到生成侧的组合性变化，也能确认理解能力没有被牺牲。
---

## 2. 课题研究目标、主要研究内容和拟解决的关键问题

### 2.1 研究目标

本课题面向统一理解与生成多模态大模型，研究低比特训练后量化的统一建模、模态与层敏感度分析、预算约束下的位宽分配以及可复现评测。考虑到硕士一年的研究周期，研究对象集中在 Janus-Pro-7B，目标是把这个新对象上的量化规律讲清楚，把方法做扎实，把结果做得可复现，不追求覆盖所有统一模型和所有硬件后端。

具体目标有四项。第一，建立统一理解与生成模型的量化配置空间，把 RTN [18]、AWQ [22]、MBQ [29] 以及模态重加权、逐通道粒度、混合精度等方案统一表示为同一组配置。第二，分析哪些模态、哪些层、哪些解码阶段对量化更敏感，解释理解侧和生成侧的不对称现象。第三，在平均位宽约束下做逐层位宽分配，并和均匀量化、手工选层在等预算下比较。第四，建立确定性评测流程，量化评测噪声，并把整个流程封装成可复现的工具包。

### 2.2 主要研究内容

（1）量化配置空间与有效性地图。把配置写成位图、粒度、激活位宽、校准配置、补偿开关和解码协议六个部分。位图可以是均匀 W3/W4，也可以是逐 block 混合；粒度可以是 per-tensor、per-channel 或 per-group；校准配置记录模态配比、样本量和是否使用教师强制；补偿开关记录视觉、理解文本、生成文本和生成图像 token 的保护项。在这个空间里逐个开关做对照，记录哪些有效、哪些无效，形成一张有效性地图。地图不仅记录正向收益，也记录零结果和负结果，方便后续研究少走弯路。

（2）模态与层敏感度分析。设计三类指标：梯度类指标统计视觉、理解文本、生成文本和生成图像 token 的梯度范数与逐通道比值；重建类指标测量逐层量化前后输出的 L2 误差和余弦误差；激活类指标统计离群值比例、动态范围和量化噪声。把三类指标归一化后组成“层乘模态”的敏感度矩阵，用来解释误差来源，也为后面的位宽分配提供依据。同时检查扩散量化文献里“输出投影和下投影更敏感”的结论在 Janus-Pro 的生成分支上是否成立。

（3）预算约束下的混合精度位宽分配。在平均位宽 b_avg 下，为 30 层语言骨干选择每层位宽，目标是让敏感度加权后的误差最小，约束是总位宽不超过 b_avg 乘以层数。这个问题可以写成多选择背包问题，用动态规划或整数规划求解。选层策略会比较均匀间隔、按敏感度比值、按原始分数和按验证集自动选择四种，评价时同时看 GenEval [36]、DPG-Bench [37] 和理解基准，避免只对单一榜单过拟合。

（4）校准集与确定性评测。校准集比较纯文本、纯理解、理解加生成混合、加入教师强制生成样本四种方案，样本量取 64、128、256 三档，观察 scale 质量随样本量的变化。评测时固定 prompt 模板、温度、随机种子、采样步数和分辨率，采用 prompt 配对和分层 bootstrap，报告 95% 置信区间。对同一个 scale 重复评测以估计噪声底，关键结论至少用三个种子复核。

（5）工具包与可选的同家族验证。把量化、校准、敏感度分析、位宽分配和评测封装成配置驱动的工具包，提供配置文件、命令行接口和结果清单。主实验在 Janus-Pro-7B 上完成。如果时间允许，再在 Janus-Pro-1B 上做同家族趋势验证，只比较 W3/W4、均匀/混合这些相对趋势，不比较绝对分数。

### 2.3 拟解决的关键问题

第一，如何在同一目标函数里平衡差异很大的模态。四类 token 的数值分布和梯度尺度不一样，简单加权会让某一模态主导 scale 搜索。本文打算用模态 mask 分离梯度，按“token 数乘每 token 平均梯度”的有效权重归一化，再做逐通道重加权，并用理解侧和生成侧的验证集约束权重。

第二，生成侧误差累积能不能观测、能不能干预。需要回答误差从哪一步开始放大、在哪些层被放大、用少量 W4 层能不能阻断。本文会按解码步和层记录量化误差、隐状态漂移和 token 置信度，画出误差传播曲线；再用同一预算下替换不同层的因果实验，区分相关性和因果性。

第三，离散位宽分配是否可解、是否稳健。位宽分配是组合优化问题，敏感度分数又有校准噪声。本文打算把问题拆成敏感度排序、预算填充和局部交换三步，引入验证集早停和多折校准，避免过拟合；对选层结果做随机扰动下的稳健性分析，报告配置方差而不是单点最优。

第四，低比特生成的评测是否可靠。生成任务方差大，小样本容易得出错误结论。本文会固定解码协议和随机种子，采用 prompt 配对和分层 bootstrap，并预先写死“全量小于 2 个百分点、子集小于 4 个百分点不解读”的规则。关键结论用全量和多种子复核，原始结果和评分脚本一并公开。

---

## 3. 拟采取的研究方法、研究方案及其可行性分析

### 3.1 研究方法

本课题以受控实验为主，配合文献调研、形式化建模、敏感度分析和组合优化。文献调研用于梳理语言模型量化 [18–27]、多模态量化 [28,29]、生成模型量化 [30–35] 和统一模型 [10–17] 的脉络。形式化建模把各种量化方案写成统一的配置元组，把位宽分配写成带约束的优化问题。受控实验在固定模型、固定校准集和固定评测协议下逐轴改变一个变量，配合配对 bootstrap 和多种子复核，尽量把方法收益和评测噪声分开。敏感度分析结合梯度、重建误差和激活离群值，用消融和因果干预验证指标和最终精度之间的关系。

### 3.2 研究方案

第一阶段是基线和评测协议，预计三个月。复现 Janus-Pro-7B 的 fp16 结果，对齐 GenEval [36]、DPG-Bench [37]、T2I-CompBench [38] 和理解基准 [39–43,49] 的评分脚本，实现确定性解码和配对 bootstrap，测量同一 scale 的重跑噪声，确定什么程度的差异才值得解读。

第二阶段是敏感度和机制分析，预计三个月。在 W8、W4、W3 三档位宽下，用 RTN [18]、AWQ [22]、MBQ [29] 和模态重加权分别生成 scale，记录逐层、逐模态、逐解码步的误差和隐状态漂移，验证理解侧和生成侧的不对称现象，定位生成敏感层。

第三阶段是混合精度和校准，预计三个月。根据敏感度结果设计效用函数，在 3.0、3.3、3.4、3.5 bit 预算下求解位宽分配，和均匀 W3/W4、线性插值、手工选层比较。同时研究校准集的模态配比、样本量和教师强制样本的影响，在 GenEval 全量、DPG-Bench 和理解基准上做配对检验。

第四阶段是验证、工具包和论文，预计三个月。完成关键消融、多种子和全量复核，时间允许的话在 Janus-Pro-1B 上做同家族趋势验证。整理工具包的配置层、命令行和结果清单，完成学位论文和投稿论文。

### 3.3 可行性分析

数据和模型方面，课题组已经有 Janus-Pro-7B 的权重、推理代码和评测脚本，GenEval、DPG-Bench、T2I-CompBench、MMMU、MME、TextVQA、MathVista、POPE 和 SEED-Bench 都已经跑通，校准数据可以使用 COCO [44] 和现有的混合图文集。

算法方面，RTN、AWQ、MBQ、逐层混合精度、梯度重加权和 bootstrap 都有现成实现或课题组已有代码。位宽分配只有 30 层，用动态规划或整数规划可以在分钟级求解，敏感度分析和混合精度也已经在预实验里跑通。

算力方面，课题组有两台服务器共 6 张 A100 80GB，本机 2 卡、副机 4 卡。GenEval 全量 2212 张图、DPG-Bench 1065 张图、T2I-CompBench 子集的一次评测可以在数小时到十几小时内完成。按 DPG 2 进程每卡、SEED 1 进程每卡的调度方式，一轮主实验大约两周可以完成，一年周期内以 7B 为主、1B 为辅是可行的。

风险主要有四个。混合精度在 DPG 上可能没有收益，甚至略有回退，预实验已经看到 GenEval 正、DPG 平的 trade-off；这种情况会如实报告，不宣称全面更优。激活 A8 可能仍然不可用，本文会把 A8 作为诊断性研究，必要时明确限定在 W4A16 和 W3A16。评测噪声可能掩盖小效应，本文会用全量、多种子和配对 bootstrap，并提前定好最小可解读差异。一年周期偏紧，本文把“7B 主结果加工具包加确定性评测”作为最小可交付集，1B 验证和打包加速作为增强项。

---

## 4. 课题的创新点

（1）目前还没有工作系统研究统一理解与生成模型的训练后量化。现有 MLLM 量化集中在理解任务，扩散量化集中在 U-Net 或 DiT 生成，本文以 Janus-Pro-7B 为对象，把理解和生成放在同一套评测里，填补这个空白。

（2）把 RTN、AWQ、MBQ、模态重加权、逐通道粒度和混合精度放进同一个配置空间，在同一协议下比较每个开关的真实收益。和只报告正向模块的做法不同，本文会同时报告零结果和负结果，比如生成文本补偿没有显著增益、T2I 和 DPG 之间存在跨榜 trade-off、A8 失效，给后续研究一个可参考的边界。

（3）系统分析统一模型里“理解侧对 W3 鲁棒、生成侧在 W3 崩溃”的不对称现象，用逐层、逐模态、逐解码步的误差传播实验解释机制。在此基础上把逐层位宽选择建模成多选择背包问题，在等平均位宽下做可解释、可自动求解的精度分配。

（4）针对生成任务方差大、跨论文结果不好比的问题，建立固定解码、prompt 配对、分层 bootstrap 和多种子复核的评测流程，实测全量和子集的噪声底。最后把这些流程封装成配置驱动的开源工具包，提供结果清单，方便复现。
---

## 5. 计划进度与预期成果

### 5.1 研究计划

课题拟用一年完成，时间为 2026 年 12 月至 2027 年 11 月。如果学校要求 11 月开题，整体安排可以前移一个月。

| 时间 | 主要工作 |
|---|---|
| 2026.12–2027.02 | 复现 Janus-Pro-7B fp16，对齐各个 benchmark 的评分脚本，实现确定性解码和配对 bootstrap，测量评测噪声 |
| 2027.03–2027.05 | 在 W8/W4/W3 下比较 RTN、AWQ、MBQ 和模态重加权，做模态与层敏感度分析，定位生成敏感层 |
| 2027.06–2027.08 | 在等平均位宽下求解混合精度分配，比较不同选层策略，研究校准集的模态配比和样本量，完成主结果 |
| 2027.09–2027.10 | 做关键消融和多种子复核，视时间在 Janus-Pro-1B 上做同家族验证，整理工具包和实验文档 |
| 2027.11 | 汇总实验结果，完成学位论文和投稿论文，准备答辩 |

### 5.2 预期成果

预期完成一套面向统一理解与生成模型的量化方法：给出理解侧和生成侧不对称的机制解释，形成模态与层敏感度画像；提出预算约束下的混合精度分配方法，在 3.0–3.5 bit 范围内给出可解释的精度与位宽权衡；建立确定性评测流程，实测全量和子集的噪声水平。工程上希望形成一套可复现的量化工具包，支持配置化量化、统一评测和结果清单。论文方面，争取发表一篇 CCF-B 或 CCF-A 类会议或期刊论文，完成硕士学位论文一篇。

### 5.3 风险与应对

主要风险是实验量偏大和跨榜结论不一致。一年时间里同时做理解、生成、混合精度和工具包，节奏会比较紧，因此把“7B 主结果、工具包、确定性评测”作为必须完成的部分，把 1B 验证和打包加速列为可选。混合精度在 DPG 上可能没有收益，这种情况会作为结论如实报告，不会为了“全面更优”去挑指标。如果 A8 仍然不可用，就把研究范围限定在 W4A16 和 W3A16，并把 A8 的失败原因作为机制分析的一部分。

---

## 6. 与本课题有关的工作积累、已有的研究工作成绩

### 6.1 前期积累

课题组已经搭建了面向 InternVL2-8B 的 W3/W4/W8 量化与评测链路，支持 RTN、AWQ、MBQ 和 MLLMQ 几种方法，打通了模型加载、逐层量化、scale 搜索、多任务评测和结果汇总。在受控复核中，W4A16 下 MLLMQ 的 MMMU 大约 0.490，方向和 MBQ 原文一致，绝对值受校准数据和评测版本影响，没有直接对齐论文。这套链路为迁移到 Janus-Pro 提供了工程基础。

Janus-Pro-7B 这边的平台也已经搭好。目前支持三模态校准，支持 RTN、AWQ、MBQ、逐通道和标量重加权、按层混合精度；已经生成 unified_384、unified_understanding_128 以及 W4A8 的 und128、mix384 等多套 scale cache；理解侧的 MMMU、MME、TextVQA、MathVista、POPE、SEED-Bench 和生成侧的 GenEval、DPG-Bench、T2I-CompBench、COCO 都已经打通。多机多卡调度、断点续跑、skip-existing、结果去重和确定性解码也已经在用。

评测方法上有一些积累。同一个 scale 重跑时，GenEval 全量的波动大约 1.63 个百分点，600 张子集约 3.83 个百分点。现在已经实现按 tag 分层、prompt 配对的 bootstrap，重复 20000 次，报告 95% 置信区间。方法比较时坚持“全量小于 2 个百分点、子集小于 4 个百分点不解读”，这为后面区分真实收益和噪声提供了依据。

### 6.2 阶段性结果

在 Janus-Pro-7B 上已经观察到比较明显的位宽不对称。3-bit RTN 在理解侧并不比学习型方法差，MMMU 大约 0.350，各方法之间差不到 2 个百分点，置信区间跨 0；但在生成侧，RTN 的 GenEval 全量只有大约 0.466，学习型方法在 0.554 到 0.584 之间，fp16 大约 0.646。DPG-Bench 上 RTN 大约 79.1，学习型方法在 82.1 到 82.7 之间，T2I-CompBench 的空间关系和数量类指标也全面下降。到 W4 和 W8，RTN 就追平了学习型方法。这说明学习型 scale 的价值主要集中在 3-bit 生成，而不是所有位宽、所有评测面。

按层混合精度也做了一轮。30 层语言骨干里选 12 层保留 W4，其余用 W3，平均位宽大约 3.40 bit。GenEval 全量大约 0.614，比均匀 W3 的 0.571 高 4.30 个百分点，95% 置信区间是 [2.19, 6.40]，大约达到 W4 精度的 96.6%，比等平均位宽的线性插值高 1.71 个百分点。逐 tag 看，收益主要来自颜色、属性绑定、位置和双物体这些组合类，和长轨迹误差累积的假设一致。不过这个收益没有迁移到 DPG-Bench，DPG 大约 81.99，和 W3 持平甚至略低；T2I-CompBench 也只是和 fp16 持平。所以混合精度存在跨榜 trade-off，不能写成全面更优。

W4A8 也做了一轮评测，包括理解侧 4 项 24 组配置和 GenEval 600 张图的 6 组配置。结果是 W4A8 比 W4A16 仍有明显退化，W3A8 在两种方法下都崩了，生成的基本是抽象纹理。三模态重加权也做了消融，修好梯度采集的 bug 之后，生成文本 token 的梯度第一次被正确采集，但 on/off 对照在 GenEval 上没有显著增益，W4 的点估计大约是 −3.33 个百分点，置信区间跨 0，W3 大约是 −0.50 个百分点；教师强制生成校准也没有显著收益。这些零结果说明，单纯加强生成信号并不能自动带来量化收益。

### 6.3 支撑条件

课题组有两台 GPU 服务器，共 6 张 A100 80GB，本机 2 卡、副机 4 卡，共享存储大约 20TB，可以放下多套 scale cache、生成图像和评测结果。Janus-Pro-7B 和 1B 的权重、COCO 以及 GenEval、DPG-Bench、T2I-CompBench 的数据和评分脚本都已经具备。申请人已经参与 Janus-Pro 的量化实验，熟悉代码、数据和评测口径，课题组在模型压缩和多模态评测方面也能提供支持。一年的周期和阶段目标匹配，主结果和工具包可以在前 9 个月完成。

## 参考文献

> 注：以下文献按正文引用顺序编号。arXiv 预印本的作者列表、卷期与页码可能与最终会议/期刊版本存在差异，正式提交前请以官方页面核对。

[1] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez A N, Kaiser Ł, Polosukhin I. Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS), 2017, 30: 5998–6008.

[2] Brown T B, Mann B, Ryder N, Subbiah M, Kaplan J, Dhariwal P, et al. Language Models Are Few-Shot Learners. Advances in Neural Information Processing Systems (NeurIPS), 2020, 33: 1877–1901.

[3] Touvron H, Lavril T, Izacard G, Martinet X, Lachaux M-A, Lacroix T, et al. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971, 2023.

[4] Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, et al. Learning Transferable Visual Models From Natural Language Supervision. International Conference on Machine Learning (ICML), 2021, PMLR 139: 8748–8763.

[5] Zhai X, Mustafa B, Kolesnikov A, Beyer L. Sigmoid Loss for Language Image Pre-Training. IEEE/CVF International Conference on Computer Vision (ICCV), 2023: 11975–11986.

[6] Alayrac J-B, Donahue J, Luc P, Miech A, Barras F, Hasson Y, et al. Flamingo: a Visual Language Model for Few-Shot Learning. Advances in Neural Information Processing Systems (NeurIPS), 2022, 35: 23716–23736.

[7] Li J, Li D, Savarese S, Hoi S. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. International Conference on Machine Learning (ICML), 2023, PMLR 202: 19730–19742.

[8] Liu H, Li C, Wu Q, Lee Y J. Visual Instruction Tuning. Advances in Neural Information Processing Systems (NeurIPS), 2023, 36.

[9] Chen Z, Wu J, Wang W, Su W, Chen G, Xing S, et al. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024: 24185–24198.

[10] Wu C, Chen X, Wu Z, Ma Y, Liu X, Pan Z, et al. Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation. arXiv:2410.13848, 2024.

[11] Chen X, Wu Z, Liu X, Pan Z, Liu W, Xie Z, Yu X, Ruan C. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling for Janus. arXiv:2501.17811, 2025.

[12] Ma Y, Liu X, Chen X, Wu Z, Pan Z, Liu W, et al. JanusFlow: Harmonizing Autoregressive and Rectified Flow for Image Generation and Understanding. arXiv:2411.07975, 2024.

[13] Sun Q, Yu Q, Cui Y, Zhang F, Zhang X, Wang Y, et al. Emu: Generative Pretraining in Multimodality. International Conference on Learning Representations (ICLR), 2024. arXiv:2307.05222.

[14] Wang X, Chen X, Zhou Z, Yuan H, Zhang Q, Wang W, et al. Emu3: Next-Token Prediction is All You Need. arXiv:2409.18869, 2024.

[15] Chameleon Team. Chameleon: Mixed-Modal Early-Fusion Foundation Models. arXiv:2405.09818, 2024.

[16] Xie J, Mao W, Bai Z, Zhang D J, Wang W, Lin K Q, et al. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation. International Conference on Learning Representations (ICLR), 2025. arXiv:2408.12528.

[17] Xie J, et al. Show-o2: Improved Native Unified Multimodal Models. arXiv:2506.15564, 2025.

[18] Jacob B, Kligys S, Chen B, Zhu M, Tang M, Howard A, Adam H, Kalenichenko D. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018: 2704–2713.

[19] Dettmers T, Lewis M, Belkada Y, Zettlemoyer L. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. Advances in Neural Information Processing Systems (NeurIPS), 2022, 35.

[20] Yao Z, Yazdani R, Cho M, Zhang M, He Y. ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. Advances in Neural Information Processing Systems (NeurIPS), 2022, 35.

[21] Frantar E, Ashkboos S, Hoefler T, Alistarh D. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. International Conference on Learning Representations (ICLR), 2023. arXiv:2210.17323.

[22] Lin J, Tang J, Tang H, Yang S, Dang X, Gan C, Han S. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. Proceedings of Machine Learning and Systems (MLSys), 2024, 6: 87–100.

[23] Xiao G, Lin J, Seznec M, Wu H, Demouth J, Han S. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. International Conference on Machine Learning (ICML), 2023, PMLR 202: 38087–38099.

[24] Shao W, Meng Y, Li L, Zhang R, Li P, Wang Y, et al. OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models. International Conference on Learning Representations (ICLR), 2024. arXiv:2308.13137.

[25] Ashkboos S, Mohtashami A, Croci M L, Li B, Cameron P, Jaggi M, et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. Advances in Neural Information Processing Systems (NeurIPS), 2024.

[26] Dettmers T, Svirschevski R, Egiazarian V, Kuznedelev D, Frantar E, Ashkboos S, et al. SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression. International Conference on Learning Representations (ICLR), 2024. arXiv:2306.03078.

[27] Egiazarian V, Panferov A, Kuznedelev D, Frantar E, Babenko A, Alistarh D. Extreme Compression of Large Language Models via Additive Quantization. International Conference on Machine Learning (ICML), 2024. arXiv:2401.06118.

[28] Wang C, Wang Z, Xu X, Tang Y, Zhou J, Lu J. Q-VLM: Post-training Quantization for Large Vision-Language Models. Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2410.08119.

[29] Zhu J, Fei Z, Fan M, Yu C, Huang J, Wei X, Hu J. Modality-Balanced Quantization for Large Vision-Language Models. arXiv:2412.19509, 2024.

[30] Li X, Liu Y, Lian L, Yang H, Dong Z, Kang D, Zhang S, Keutzer K. Q-Diffusion: Quantizing Diffusion Models. IEEE/CVF International Conference on Computer Vision (ICCV), 2023: 17535–17545.

[31] Peebles W, Xie S. Scalable Diffusion Models with Transformers. IEEE/CVF International Conference on Computer Vision (ICCV), 2023: 4195–4205.

[32] Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B. High-Resolution Image Synthesis with Latent Diffusion Models. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022: 10684–10695.

[33] Ho J, Jain A, Abbeel P. Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems (NeurIPS), 2020, 33: 6840–6851.

[34] van den Oord A, Vinyals O, Kavukcuoglu K. Neural Discrete Representation Learning. Advances in Neural Information Processing Systems (NeurIPS), 2017, 30: 6306–6315.

[35] Esser P, Rombach R, Ommer B. Taming Transformers for High-Resolution Image Synthesis. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021: 12873–12883.

[36] Ghosh D, Hajishirzi H, Schmidt L. GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2310.11513.

[37] Hu X, Wang R, Fang Y, Fu B, Cheng P, Yu G. ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment. arXiv:2403.05135, 2024.

[38] Huang K, Sun K, Xie E, Li Z, Liu X. T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation. NeurIPS Datasets and Benchmarks Track, 2023. arXiv:2307.06350.

[39] Yue X, Ni Y, Zhang K, Zheng T, Liu R, Zhang G, et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024: 9556–9567.

[40] Fu C, Chen P, Shen Y, Qin Y, Zhang M, Lin X, et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv:2306.13394, 2023.

[41] Singh A, Natarajan V, Shah M, Jiang Y, Chen X, Batra D, Parikh D, Rohrbach M. Towards VQA Models That Can Read. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019: 8317–8326.

[42] Lu P, Bansal H, Xia T, Liu J, Li C, Hajishirzi H, et al. MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. International Conference on Learning Representations (ICLR), 2024. arXiv:2310.02255.

[43] Li Y, Du Y, Zhou K, Wang J, Zhao W X, Wen J-R. Evaluating Object Hallucination in Large Vision-Language Models. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. arXiv:2305.10355.

[44] Lin T-Y, Maire M, Belongie S, Hays J, Perona P, Ramanan D, Dollár P, Zitnick C L. Microsoft COCO: Common Objects in Context. European Conference on Computer Vision (ECCV), 2014: 740–755.

[45] Hessel J, Holtzman A, Forbes M, Le Bras R, Choi Y. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.

[46] Kirstain Y, Polyak A, Singer U, Matiana S, Penna J, Levy O. Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation. Advances in Neural Information Processing Systems (NeurIPS), 2023.

[47] Wu X, Hao Y, Sun K, Chen Y, Zhu F, Zhao R, Li H. Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv:2306.09341, 2023.

[48] Xu J, Liu X, Wu Y, Tong Y, Li Q, Ding M, et al. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. Advances in Neural Information Processing Systems (NeurIPS), 2023.

[49] Li B, Wang R, Wang G, Ge Y, Ge Y, Shan Y, et al. SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension. arXiv:2307.16125, 2023.
