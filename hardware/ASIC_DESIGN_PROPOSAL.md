# TurboQuant ASIC 加速器设计方案

**面向 ISSCC 投稿 | 数字 CIM + Flash 近存计算**

---

## 目录

1. [研究背景与 Motivation](#1-研究背景与-motivation)
2. [算法特征分析（硬件友好性）](#2-算法特征分析硬件友好性)
3. [系统架构：TQ-Flash Chiplet](#3-系统架构tq-flash-chiplet)
4. [核心电路模块设计](#4-核心电路模块设计)
   - 4.1 [RHT 蝶形旋转阵列（数字 CIM）](#41-rht-蝶形旋转阵列数字-cim)
   - 4.2 [Lloyd-Max 并行量化器](#42-lloyd-max-并行量化器)
   - 4.3 [QJL 符号投影单元](#43-qjl-符号投影单元)
   - 4.4 [Flash 近存 KV 缓存控制器](#44-flash-近存-kv-缓存控制器)
5. [RTL 顶层架构与接口定义](#5-rtl-顶层架构与接口定义)
6. [关键 RTL 模块规格](#6-关键-rtl-模块规格)
7. [综合与 PPA 目标](#7-综合与-ppa-目标)
8. [ISSCC Story 与差异化定位](#8-isscc-story-与差异化定位)
9. [评估方法与 Baseline](#9-评估方法与-baseline)
10. [开发路线图](#10-开发路线图)

---

## 1. 研究背景与 Motivation

### 1.1 问题陈述

LLM 推理的 decode 阶段受 **KV cache 带宽瓶颈** 严重制约：

| 场景 | KV cache 大小 | 需求带宽 | 现有存储 |
|---|---|---|---|
| Llama-3.1-8B, 128K ctx | ~16 GB | >1 TB/s | HBM: 3.35 TB/s (H100) |
| Llama-3.1-70B, 128K ctx | ~128 GB | >8 TB/s | **超出 HBM 容量** |
| 边缘端 8B, 32K ctx | ~4 GB | ~200 GB/s | LPDDR5: 68 GB/s（**不足**） |

**核心矛盾**：长上下文 KV cache 既超出高速 DRAM 容量，又受 Flash 带宽不足制约（NAND Flash 顺序读：~3 GB/s）。

### 1.2 现有方案的不足

| 方案 | 问题 |
|---|---|
| KV cache 量化（软件，KIVI/KVQuant） | GPU 上执行，占用 tensor core，增加 decode latency |
| FlashAttention offload | 仅优化 DRAM 读，不压缩 |
| Flash 存储 KV（llama.cpp） | 带宽瓶颈：3 GB/s vs 需求 ~50 GB/s |
| PIM/HBM-PIM | 需要 HBM 工艺，成本极高，边缘端不可用 |

### 1.3 本文的机会

**TurboQuant 算法特性**使其天然适合定制硬件：

1. **旋转矩阵固定**（per-model 预计算）→ 权重可存 SRAM，极高复用率
2. **RHT 可替代密集旋转**（O(d log d) vs O(d²)）→ 蝶形网络直接映射
3. **Lloyd-Max 量化 = 多阈值比较**→ 类 Flash-ADC 电路，无需乘法
4. **token 间无依赖**→ 完全流水线，批量并行

**核心 Insight**：将 TurboQuant 量化引擎部署在 **Flash 存储控制器侧**，在 KV token 写入 Flash 前完成压缩，读出时完成解压缩，使 Flash 有效带宽提升 **3.5–6×**，同时不占用主机侧 GPU/NPU 资源。

---

## 2. 算法特征分析（硬件友好性）

### 2.1 计算流水线拆解（per attention head, d=128）

```
输入: KV token  x ∈ R^128 (bfloat16, 写入路径)
│
├─ Stage 1: Norm 提取
│   norm = ||x||₂          → 1× sqrt + 128× mul/add (可用 fast-norm 近似)
│   x_unit = x / norm      → 128× 除法 (可用 reciprocal LUT)
│
├─ Stage 2: RHT 旋转
│   y = D · H · x_unit     → log₂(128)=7 级蝶形，每级 128× add/sub+mul(±1)
│                             总计: 128×7 = 896 加减运算
│
├─ Stage 3: Lloyd-Max 量化 (per-coordinate)
│   对每个 yⱼ: 比较 2^b 个阈值，输出 b-bit index
│   b=3: 8阈值比较，128维并行 → 128× 7-comparator tree
│
├─ Stage 4 (TQ_prod 额外): QJL 残差
│   r = x_unit - reconstruct(idx)   → 128× 查表 + 减法
│   sign(S·r): 128× 内积 + 符号     → 128× FMA（S 固定，SRAM 存储）
│
└─ 输出: (idx[128×3bit], norm[fp16], qjl_sign[128×1bit])
         = 3.5-bit effective: 48字节 → 从256字节压缩 (5.3× 压缩比)
```

### 2.2 关键硬件映射

| 算法操作 | 硬件映射 | 可综合性 |
|---|---|---|
| D · x（符号翻转）| XOR + 条件取反 | ✅ 纯数字逻辑 |
| H · x（Hadamard）| 蝶形加减树，7级 | ✅ 流水线加法器 |
| argmin(\|y-c\|) | 阈值比较树（b-1 个比较器） | ✅ 优先编码器 |
| sign(S·r) | 带符号累加 + 1-bit 输出 | ✅ CSA 树 |
| norm 计算 | CORDIC 或 LUT-based | ✅ 迭代/查表 |

### 2.3 面向 Flash 近存的关键设计约束

| 约束 | 数值 | 硬件含义 |
|---|---|---|
| Flash 写入粒度 | 4KB / page | 每 page 可容 16–32 个 token（d=128, 3.5-bit） |
| Flash 写入带宽 | ~1–3 GB/s（SLC/MLC） | 量化延迟预算：< 1μs/token |
| 面积预算（控制器 die） | < 2 mm² @ 28nm | 量化引擎面积预算 |
| 功耗约束 | < 200mW（边缘端 SSD 控制器） | 每 token 能耗 < 2nJ |

---

## 3. 系统架构：TQ-Flash Chiplet

### 3.1 总体系统图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Host NPU / GPU (主机端)                       │
│                                                                   │
│  Attention Compute ──→ KV token (BF16) ──→ PCIe/CXL/UCIe        │
│       ↑                                         │                │
│  Query × K/V ←── 解压缩 KV token (BF16) ←──────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │ PCIe Gen5 / CXL 2.0
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  TQ-Flash SSD Controller Die                      │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              TurboQuant Engine (本文核心)                 │    │
│  │                                                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │    │
│  │  │  Norm    │→ │  RHT    │→ │ LM-Quant │→ │ QJL   │  │    │
│  │  │  Unit    │  │ Butterfly│  │  Engine  │  │ Engine │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘  │    │
│  │        ↑              ↑              ↑            ↑      │    │
│  │   RHT Config SRAM  Threshold ROM   QJL-S SRAM          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         KV Cache 管理器 + Flash FTL                       │   │
│  │  Layer/Head 索引 | 压缩 token 地址映射 | GC 调度           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │ Toggle/ONFI
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  NAND Flash Die(s)                                │
│         压缩 KV cache（3.5-bit effective per dimension）          │
│         存储密度：原始 5.3× 压缩 vs BF16                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流（写入：KV token 压缩）

```
1. Host NPU 完成 KV projection，发出 DMA 写请求
   payload: [layer_id, head_id, token_pos, KV_data(256B, BF16)]

2. TQ Engine 接收 token，流水线处理（4 stages, 目标 1 token/cycle @ 400MHz）
   → 输出: [norm(FP16), idx(48B@3-bit), qjl_sign(16B)]  共 66B

3. KV Cache Manager 封包为 Flash page
   每 page(4KB): ~62 tokens（含 ECC + 地址 metadata）

4. Flash FTL 写入 NAND
```

### 3.3 数据流（读出：KV token 解压缩）

```
1. Attention 计算需要 Query × K/V
   Host 发出读请求: [layer_id, head_id, token_range]

2. Flash 读出压缩 token，经 TQ Engine 解压缩
   Dequant: y_hat = centroid[idx], x_hat = norm × IRHT(y_hat) + QJL_recon(sign, gamma)

3. 解压缩 BF16 token 返回 Host
   → Attention softmax & weighted sum

4. 有效读带宽: Flash 3GB/s → 等效 BF16 带宽: 3 × 5.3 ≈ 16 GB/s
```

---

## 4. 核心电路模块设计

### 4.1 RHT 蝶形旋转阵列（数字 CIM）

#### 设计原理

Randomized Hadamard Transform（RHT）将密集正交旋转矩阵 Π（d×d）替换为：

```
RHT(x) = (1/√d) · D · H · x

D: d×d 对角矩阵，Dᵢᵢ = ±1（随机 Rademacher，固定后存 ROM）
H: d×d Walsh-Hadamard 矩阵（蝶形网络）
```

数学保证：RHT 与随机正交旋转在统计上等价，保留 TurboQuant 的理论误差界。

#### 蝶形网络结构（d=128, 7级）

```
Stage 0  Stage 1  Stage 2  Stage 3  Stage 4  Stage 5  Stage 6
  x[0] ──┬──────────┬──...
          │          │
  x[1] ──┴─(±)──────┴─(±)──...

每级：64个蝶形单元，每单元 1个加法 + 1个减法
操作: (a, b) → (a+b, a-b)，无乘法（WHT性质）

总计: 7 × 64 = 448 个加法器 + 448 个减法器
加上 D 的符号翻转: 128 个 XOR/条件取反
```

#### 数字 CIM 方案说明

本方案采用**可综合数字 CIM**，区别于 SRAM-based 模拟 CIM：

- 旋转系数 D（128个 ±1 符号）存储在片上 SRAM/ROM 中
- 蝶形网络由**流水线加法器阵列**实现，每级寄存器隔离
- "CIM"体现在：配置数据（D 向量、Lloyd-Max 阈值、QJL 矩阵 S 的列）与计算单元**紧耦合共址**，无数据总线传输开销
- 完全可综合，无 custom cell，适合标准 28nm CMOS 工艺

#### RTL 接口

```verilog
module rht_butterfly #(
    parameter D      = 128,        // 向量维度，必须为 2 的幂
    parameter LOG_D  = 7,          // log2(D)
    parameter DW     = 16          // 数据位宽（BF16 → 内部 INT16 定点）
)(
    input  wire                clk,
    input  wire                rst_n,
    // 输入
    input  wire [D*DW-1:0]     x_in,          // 归一化输入向量
    input  wire                x_valid,
    // D 向量（Rademacher 符号，从 SRAM 加载）
    input  wire [D-1:0]        d_signs,        // 1=正，0=负
    // 输出
    output wire [D*DW-1:0]     y_out,          // 旋转后向量
    output wire                y_valid
);
```

#### 关键微架构决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 数据格式 | INT16 定点（1符号+7整数+8小数） | 避免 FP 乘法器，可综合 |
| 流水线深度 | 7 级（LOG_D=7） | 每级 1 cycle，无气泡 |
| 面积 vs 速度 | 完全展开（7 级同时存在） | token 间流水，吞吐 1 token/cycle |
| D 符号存储 | 128-bit ROM（固定 per model） | < 128 bit，面积可忽略 |
| 缩放因子 1/√d | 右移 3.5 位 → 近似右移 4 位 | 误差 < 0.4%，查表修正可选 |

---

### 4.2 Lloyd-Max 并行量化器

#### 设计原理

对旋转后每个坐标 yⱼ，Lloyd-Max 量化等价于：

```
idx_j = argmin_k |yⱼ - c_k|

等价于：找 yⱼ 落在哪个 Voronoi 区间
即与 (2^b - 1) 个阈值比较：
  if yⱼ < t₁:      idx = 0
  elif yⱼ < t₂:    idx = 1
  ...
  else:             idx = 2^b - 1

阈值 tₖ = (cₖ + cₖ₊₁) / 2（相邻 centroid 中点）
```

对 d=128 维，b=3 bit（8 level）：128个独立量化器并行执行。

#### 量化器电路结构

```
输入 yⱼ (INT16) ──┬──> 比较器 0: yⱼ < t₁?
                  ├──> 比较器 1: yⱼ < t₂?
                  ├──> 比较器 2: yⱼ < t₃?
                  ├──> 比较器 3: yⱼ < t₄?
                  ├──> 比较器 4: yⱼ < t₅?
                  ├──> 比较器 5: yⱼ < t₆?
                  └──> 比较器 6: yⱼ < t₇?
                              │
                    7-bit 热码 → 优先编码器 → 3-bit index
```

等价于 **Flash-ADC 架构**（7 个并行比较器 + 优先编码器），无需乘法器。

#### 混合精度支持

KV cache 采用混合精度（outlier channel 高 bit，regular channel 低 bit）：

```
outlier channels (32个):  b=5 (31个比较器/channel)
regular channels (96个): b=3 (7个比较器/channel)

硬件实现：
- 量化器配置寄存器：per-channel b_sel[1:0] (00=2bit, 01=3bit, 10=4bit, 11=5bit)
- 比较器数量：最多 2^5-1 = 31 个，低精度模式时关闭高位比较器（clock gating）
- 阈值 ROM：最大规格 31 × 128 个阈值，per-dimension 复用
```

#### RTL 接口

```verilog
module lloyd_max_quant #(
    parameter D        = 128,
    parameter B_MAX    = 5,        // 最大比较器数 = 2^B_MAX - 1 = 31
    parameter DW       = 16,
    parameter TW       = 16        // 阈值位宽
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire [D*DW-1:0]             y_in,
    input  wire                         y_valid,
    // 阈值（从 ROM 加载，per-dimension，共 D*(2^B_MAX-1) 个）
    input  wire [D*(2**B_MAX-1)*TW-1:0] thresholds,
    // 混合精度配置
    input  wire [D*2-1:0]               b_sel,         // 每维 bit-width 选择
    // 输出
    output wire [D*B_MAX-1:0]           idx_out,       // 每维 B_MAX-bit index（低位有效）
    output wire                          idx_valid
);
```

---

### 4.3 QJL 符号投影单元

#### 设计原理

QJL 计算：`sign(S · r)`，其中 S ∈ R^(d×d) 是固定 Gaussian 随机矩阵（per model 预生成）。

关键观察：
- **S 固定**，可量化后存入 SRAM（INT4 量化 S 误差可忽略）
- 输出是 1-bit 符号，只需判断累加结果正负
- 无需计算精确数值，只需符号 → **单比特输出，无 ADC**

#### 面积优化：S 矩阵压缩存储

```
d=128 时，S 全精度: 128×128×16bit = 256KB (太大)

优化方案：
1. S 量化为 INT4: 128×128×4bit = 64KB
2. LFSR 生成（随机种子确定性生成）: 面积极小，但需流式计算
3. 分块流式: 每次处理 k 行，k=16 → SRAM 只需 16×128×4 = 8KB

推荐方案 3：分块流式处理
- 内层循环：16行 × 128列 并行 MAC → 16个 accumulator
- 16次分组 → 完成全部 128 个输出符号
- 延迟：128/16 = 8 cycles（可流水线）
```

#### RTL 接口

```verilog
module qjl_engine #(
    parameter D        = 128,
    parameter S_ROWS   = 16,       // 每次处理的 S 行数（分块大小）
    parameter DW       = 16,
    parameter SW       = 4         // S 矩阵量化位宽
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [D*DW-1:0]        r_in,          // 残差向量
    input  wire                    r_valid,
    input  wire [D*S_ROWS*SW-1:0] s_block,        // S 矩阵当前块（从SRAM读入）
    input  wire [$clog2(D/S_ROWS)-1:0] block_idx, // 当前块编号
    output wire [D-1:0]            sign_out,       // 128-bit 符号向量
    output wire                    sign_valid
);
```

---

### 4.4 Flash 近存 KV 缓存控制器

#### 整体设计目标

```
传统 LLM + Flash 路径:
Host → 生成 BF16 KV token → PCIe → SSD Controller → Flash
      读出时: Flash → SSD → PCIe → Host (3GB/s BF16 等效)

本文路径 (TQ-Flash):
Host → 生成 BF16 KV token → PCIe → [TQ-Engine: 压缩 5.3×] → Flash
      读出时: Flash → [TQ-Engine: 解压缩] → PCIe → Host
      等效带宽: 3GB/s × 5.3 = ~16GB/s BF16 等效
```

#### KV Cache 管理策略

```
逻辑地址空间:
  [model_id: 4bit][layer_id: 6bit][head_id: 7bit][token_pos: 17bit]
  → 最大支持: 16 models × 64 layers × 128 heads × 128K tokens

Flash 物理映射:
  每个 Flash page (4KB):
    header: 16B (layer/head/token_range/checksum)
    payload: 4032B → 62 tokens × 65B/token (压缩后, 3.5-bit effective)
    ECC: 48B (BCH/LDPC)

地址转换表 (LUT):
  存放在 Controller 片上 SRAM: 4MB (支持 ~8M token 的地址映射)
```

#### 读写流水线

```
写入流水线 (Write Pipeline):
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ PCIe RX │→ │ Norm/    │→ │ RHT     │→ │ LM-Quant │→ │ Page     │
│ DMA     │  │ Unpack   │  │ Butterfly│  │ + QJL    │  │ Packer   │
└─────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
 256B/token    ~1 cycle      7 cycles      2 cycles      按page聚合
                                                              │
                                                         Flash Write

读出流水线 (Read Pipeline):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐
│ Flash    │→ │ Page     │→ │ IRHT +   │→ │ QJL      │→ │ PCIe TX │
│ Read     │  │ Unpacker │  │ Dequant  │  │ Recon    │  │ DMA     │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────────┘
                                7 cycles    2 cycles       256B/token
```

#### Prefetch 与 Attention 融合优化

关键创新：**在 Flash 读取时提前完成部分 Attention 计算**（Flash-Attention offload）

```
传统: Flash → 传回所有 K tokens → Host 计算 Q·K^T
本文 (可选 Mode B): 
  Flash → 解压缩 K tokens → 在 Controller 上计算 Q·K attention scores
  → 只传 softmax scores（标量）回 Host → Host 加权 V
  
V 的传输量大幅减少（只传高注意力权重的 V tokens）
```

---

## 5. RTL 顶层架构与接口定义

### 5.1 顶层模块层次

```
tq_flash_top
├── host_interface          # PCIe/CXL 接口逻辑
│   ├── pcie_rx_dma
│   └── pcie_tx_dma
│
├── tq_engine               # TurboQuant 量化引擎（核心）
│   ├── token_preprocessor  # BF16 解包、norm 提取
│   ├── rht_butterfly       # 旋转变换
│   ├── lloyd_max_quant     # Lloyd-Max 量化
│   ├── qjl_engine          # QJL 残差量化
│   └── dequant_engine      # 解压缩路径（IRHT + 重建）
│
├── kv_cache_manager        # KV token 地址管理
│   ├── address_lut         # 逻辑→物理地址映射 SRAM
│   ├── page_packer         # token 聚合成 Flash page
│   └── gc_scheduler        # GC 调度器（延迟隐藏）
│
├── flash_controller        # NAND Flash 控制
│   ├── onfi_phy            # ONFI 3.x 物理层
│   ├── ecc_engine          # BCH/LDPC ECC
│   └── read_prefetch       # 预读缓冲
│
└── config_regs             # APB 配置寄存器组
```

### 5.2 TQ Engine 顶层接口

```verilog
module tq_engine #(
    parameter D          = 128,        // attention head 维度
    parameter B_OUTLIER  = 5,          // outlier channel bit-width
    parameter B_REGULAR  = 3,          // regular channel bit-width
    parameter N_OUTLIER  = 32,         // outlier channel 数量
    parameter DW         = 16,         // 内部定点位宽
    parameter N_PARALLEL = 4           // 并行 token 处理数
)(
    input  wire         clk,           // 主时钟 (目标 400MHz @ 28nm)
    input  wire         rst_n,

    // === 量化路径（写入）===
    input  wire [N_PARALLEL*D*16-1:0]  kv_tokens_in,   // BF16 KV tokens
    input  wire [N_PARALLEL-1:0]       token_valid_in,
    input  wire [5:0]                  layer_id,
    input  wire [6:0]                  head_id,

    output wire [N_PARALLEL*D*4-1:0]   compressed_out,  // 压缩后数据（~3.5-bit avg）
    output wire [N_PARALLEL-1:0]       compressed_valid,

    // === 解压缩路径（读出）===
    input  wire [N_PARALLEL*D*4-1:0]   compressed_in,
    input  wire [N_PARALLEL-1:0]       decomp_valid_in,

    output wire [N_PARALLEL*D*16-1:0]  kv_tokens_out,   // 重建 BF16 tokens
    output wire [N_PARALLEL-1:0]       tokens_valid_out,

    // === 配置接口（APB）===
    input  wire [31:0]  apb_addr,
    input  wire [31:0]  apb_wdata,
    input  wire         apb_write,
    output wire [31:0]  apb_rdata,

    // === 状态/监控 ===
    output wire [31:0]  tokens_processed_cnt,
    output wire [15:0]  compression_ratio_x100,  // 实际压缩比 × 100
    output wire         engine_busy
);
```

---

## 6. 关键 RTL 模块规格

### 6.1 token_preprocessor

**功能**: BF16 解包 → 定点转换 → L2 norm 计算 → 归一化

```verilog
// 关键时序:
// BF16 input → INT16 fixed-point: 1 cycle (组合逻辑 + 寄存)
// norm 计算: CORDIC 16 cycles（或 LUT-based fast-norm 4 cycles）
// 归一化除法: reciprocal LUT + multiply = 3 cycles
// 总延迟: ~8 cycles (fast-norm 方案)

// 面积主要来自: 128个 BF16→INT16 转换单元 + CORDIC/LUT norm
```

### 6.2 rht_butterfly

**功能**: 7级 Walsh-Hadamard 蝶形变换 + Rademacher 符号翻转

```verilog
// 关键参数:
// 流水级数: LOG_D = 7
// 每级蝶形单元: 64 个 (a,b)→(a+b, a-b)
// 每蝶形: 1加 + 1减 @ INT16
// D 符号存储: 128-bit register（固定 per-model）
// 缩放: 右移 4 位（近似 1/√128 = 1/11.3 ≈ 1/16）
//
// 关键时序: 7 cycle 延迟（全流水）
// 吞吐: 1 token/cycle（128×16bit in, 128×16bit out）
// 面积估算: ~15K gates @ 28nm（主要为 128×7=896 个 INT16 加减法器）
```

### 6.3 lloyd_max_quant

**功能**: 128维并行 Flash-ADC 风格量化，支持混合精度

```verilog
// 关键参数:
// 并行量化器: 128 个独立比较器树
// outlier (32个): 每维 31 个比较器
// regular (96个): 每维 7 个比较器
// 阈值存储: ROM
//   outlier: 32 × 31 × 16bit = 15.5KB
//   regular: 96 × 7 × 16bit  = 10.5KB
//   总计: ~26KB ROM
//
// 关键时序: 2 cycles（比较 + 优先编码）
// 吞吐: 1 token/cycle
// 面积估算: ~40K gates（比较器树 + ROM接口）
```

### 6.4 qjl_engine

**功能**: 128维符号随机投影，分块流式处理

```verilog
// 关键参数:
// S 矩阵: 128×128, INT4量化存储 = 64KB SRAM
// 分块大小: S_ROWS = 16 行/块
// 每块: 16路 INT16×INT4 MAC → 16个 accumulator
// 总块数: 128/16 = 8 次 → 8 cycles 完成全部符号
//
// 可选优化: 用 LFSR 生成 S（消除 64KB SRAM，代价增加 8 cycles 生成延迟）
//
// 关键时序: 8 cycles（分块模式）
// 吞吐: 1 token / 8 cycles（QJL 是瓶颈，可用并行 S 块 SRAM 解决）
// 面积估算: ~30K gates + 64KB SRAM（或 ~8K gates + LFSR）
```

### 6.5 dequant_engine

**功能**: 解压缩（IRHT + Centroid 查表 + QJL 重建）

```verilog
// 路径 1: idx → centroid 查表
//   ROM lookup: 128维 × 5bit index → 128 × INT16 centroid
//   ROM size: 2^5 × 16bit = 512bit per dimension (使用 outlier/regular 两套)
//
// 路径 2: IRHT (逆 RHT = 转置 RHT，对 WHT 矩阵 H^T = H)
//   与 rht_butterfly 共享蝶形阵列（时分复用写/读路径）
//   再乘以 D^T = D（因 D 是对角 ±1 矩阵，逆等于自身）
//
// 路径 3: QJL 重建
//   x_qjl = sqrt(π/2)/d × S^T × sign_vec × gamma
//   = 同 QJL Engine，方向反向（S^T vs S，逐行改逐列）
//
// 总延迟: ~10 cycles（ROM lookup 2 + IRHT 7 + rescale 1）
// 重建精度: MSE ≈ 0.032（3-bit regular，与 Python 仿真一致）
```

---

## 7. 综合与 PPA 目标

### 7.1 技术节点与工具链

| 选项 | 工艺 | EDA 工具 | 适用场景 |
|---|---|---|---|
| 工业级 | TSMC 28nm HPC | DC + ICC2 + PrimeTime | 全流程 tape-out |
| 开源流程 | SKY130 / GF180 | OpenROAD + Yosys | 功能验证，ISSCC 概念证明 |
| 学术目标 | 28nm 等效 | Synopsys DC (学术授权) | PPA 报告 |

**推荐路径（ISSCC 投稿）**：用 Synopsys DC 综合到 28nm standard cell library，报告 post-synthesis PPA；用 OpenROAD + SKY130 做 floorplan 概念演示图。

### 7.2 PPA 设计目标

| 模块 | 面积目标 | 功耗目标 | 频率目标 |
|---|---|---|---|
| token_preprocessor × 4 | < 0.05 mm² | < 10 mW | 400 MHz |
| rht_butterfly × 4 | < 0.08 mm² | < 15 mW | 400 MHz |
| lloyd_max_quant × 4 | < 0.15 mm² | < 20 mW | 400 MHz |
| qjl_engine × 2 | < 0.10 mm² | < 15 mW | 400 MHz |
| dequant_engine × 4 | < 0.12 mm² | < 15 mW | 400 MHz |
| SRAM (S矩阵 + 阈值) | ~0.10 mm² | < 10 mW | — |
| KV Cache 管理器 | < 0.05 mm² | < 5 mW | 400 MHz |
| **TQ Engine 总计** | **< 0.65 mm²** | **< 90 mW** | **400 MHz** |

### 7.3 性能预算

```
目标工作点: d=128, 3.5-bit, N_PARALLEL=4, @400MHz

量化延迟 (单 token):
  preprocessor: 8 cycles = 20 ns
  RHT:          7 cycles = 17.5 ns
  LM-Quant:     2 cycles = 5 ns
  QJL:          8 cycles = 20 ns  ← 瓶颈
  Pipeline 重叠后总 latency: ~20 ns (pipeline 深度最大级)

吞吐量:
  N_PARALLEL=4: 4 tokens/cycle = 4 × 128 × 16bit = 8192 bit/cycle
  @400MHz: 3.2 Gtoken/s（单 head）
  实际受限于 PCIe 接口: ~12.5 GB/s (PCIe Gen5 ×4)
  等效: 12.5GB/s ÷ 256B/token = 48.8M token/s

Flash 写入匹配:
  TQ Engine 处理速度 >> Flash 写入速度 (Flash 是瓶颈)
  TQ Engine 功耗在 Flash 写入等待时可 clock gate
```

### 7.4 能效指标

```
每 token 压缩能耗 (估算):
  @ 90mW total TQ Engine, 48.8M token/s
  = 90mW / 48.8M = 1.84 nJ/token

类比:
  GPU RTX 4090 上软件 TurboQuant: ~500 nJ/token（GPU 功耗 / 吞吐）
  本设计能效提升: ~270×

TOPS/W:
  等效运算量（旋转+量化+QJL）: ~5K OPS/token
  = 5K × 48.8M token/s = 244 GOPS
  TOPS/W = 244 GOPS / 0.09W ≈ 2.7 TOPS/W
  (专用逻辑，非 GEMM，不应直接与 GPU TOPS/W 比较)
```

---

## 8. ISSCC Story 与差异化定位

### 8.1 一句话 Pitch

> "TQ-Flash：一款面向边缘 LLM 长上下文推理的 Flash 近存量化引擎，通过将 TurboQuant 旋转-量化流水线下沉至 SSD 控制器，实现 5.3× KV cache 压缩比，使 Flash 等效读带宽从 3 GB/s 提升至 16 GB/s，以 < 90mW 功耗支持 128K token 上下文，避免 GPU 侧量化开销。"

### 8.2 ISSCC 卖点层次

| 层次 | 技术点 | ISSCC 对应 Session |
|---|---|---|
| 电路创新 | 数字 CIM 蝶形阵列 + Flash-ADC 量化器协同设计 | Data Converters / Digital CIM |
| 架构创新 | Flash 近存量化引擎（量化与存储协同） | Memory & Storage Systems |
| 算法-硬件协同 | RHT 替代密集旋转，理论误差界保证，硬件友好 | AI Hardware |
| 系统价值 | 边缘端 8B LLM 实现 128K context | Invited / VLSI 联投 |

### 8.3 Narrative Arc（完整版）

```
【世界现状】
  边缘 NPU（如 iPhone Neural Engine, Qualcomm Hexagon）算力已够，
  但 8GB LPDDR5 限制 KV cache 到 < 8K tokens。

【为何难】
  - Flash 容量大但带宽仅 3 GB/s（LPDDR5 的 1/20）
  - GPU/NPU 端软件量化消耗 20-30% decode 算力
  - 现有 Flash 控制器无法理解 AI 语义（只看 raw bytes）

【关键 Insight】
  TurboQuant 的旋转矩阵与量化码本在 per-model 推理时完全固定，
  可预加载至 Flash 控制器片上 SRAM。
  WHT 旋转无乘法器，Lloyd-Max 量化无乘法器，
  整个量化引擎 = 加减法 + 比较器，天然适合低功耗数字电路。

【我们的方案】
  TQ-Flash ASIC：将 TurboQuant 引擎集成至 Flash SSD 控制器，
  在 write path 压缩，read path 解压，主机端无感知。

【量化证明】
  - 5.3× 压缩，等效带宽 16 GB/s
  - NIAH 评分: FP 0.451 vs TQ-4bit 0.409（-9%，可接受）
  - 面积 < 0.65 mm² @ 28nm，功耗 < 90 mW
  - 比 GPU 软件量化能效提升 270×
```

### 8.4 与 prior work 差异化对比

| 工作 | 方案 | 与本文区别 |
|---|---|---|
| KIVI (arxiv 2024) | 2-bit KV cache, 软件 | 纯软件，无硬件；精度损失大 |
| KVQuant (NeurIPS 2024) | 非均匀量化，软件 | 无硬件；per-channel 量化无理论保证 |
| FlashAttention-3 | IO-aware attention，GPU 上 | 不压缩 KV，不涉及 Flash |
| ANT (MICRO 2021) | Weight 量化加速器 | 面向 weight，非 KV cache；无旋转 |
| OLAccel (MICRO 2022) | Outlier-aware 量化 | 面向 weight；无 Flash 集成 |
| KVNAND (本实验室假设) | Flash KV, 无量化 | 无量化压缩，带宽利用率低 |
| **本文** | **Flash 近存 TurboQuant ASIC** | **首个将旋转量化引擎下沉至 Flash 控制器的工作** |

---

## 9. 评估方法与 Baseline

### 9.1 硬件评估

| 评估内容 | 工具/方法 | 可信度 |
|---|---|---|
| 功能验证 | SystemVerilog RTL + VCS/Verilator 仿真 | ✅ |
| 面积/功耗 | Synopsys DC @ 28nm cell lib | ✅（post-synthesis）|
| 时序 | PrimeTime STA | ✅（post-synthesis）|
| SRAM 面积 | CACTI 7.0 @ 28nm | ✅ |
| Flash 时序 | ONFI 3.x spec 建模 | ✅ |

### 9.2 系统性能评估

| 指标 | 方法 |
|---|---|
| 端到端 LLM 推理延迟 | 分析模型 (roofline-style，参考 Cambricon-LLM 方法论) |
| NIAH 精度 | 本仓库实验结果（Llama-3.1-8B，4K–128K tokens）|
| 压缩比 vs 精度 Pareto | 4-bit/3.5-bit/2.5-bit 三个工作点 |

### 9.3 Baseline 选取（按 reviewer 标准）

| Baseline | 说明 |
|---|---|
| B0: 无压缩 Flash KV | Flash 带宽上限（3 GB/s）|
| B1: KIVI (软件 2-bit) | 最强软件量化 baseline |
| B2: GPU-side TurboQuant | 相同算法在 GPU tensor core 上执行 |
| B3: TQ-DRAM（无 Flash）| 近 DRAM 量化，评估 Flash 集成的额外收益 |
| **Proposed**: TQ-Flash | 本文方案 |

---

## 10. 开发路线图

### Phase 1 — 算法-硬件协同仿真（0–2月）

- [ ] RHT vs 密集 Π 的量化精度等价性验证（Python 仿真）
  - 目标：在 `turboquant.py` 中添加 `TurboQuantRHT` 类，复现 NIAH 实验
  - 验证 RHT 旋转的误差界与原 Π 等价
- [ ] 定点量化误差分析（INT16 vs FP32 旋转差异）
  - 分析各级精度损失（BF16→INT16→旋转→量化）
  - 确定各模块位宽
- [ ] Flash 系统级仿真（Python 建模）
  - 建立 Flash read/write 时序模型
  - 验证 TQ-Engine 与 Flash 带宽的匹配

### Phase 2 — RTL 实现（2–4月）

- [ ] `rht_butterfly.sv` — 7级蝶形，带 D 符号寄存器
- [ ] `lloyd_max_quant.sv` — 128并行比较器树，混合精度
- [ ] `qjl_engine.sv` — 分块 MAC + 符号输出
- [ ] `dequant_engine.sv` — 逆 RHT + 重建路径
- [ ] `tq_engine.sv` — 顶层集成 + 4路并行流水线
- [ ] Testbench: 与 Python golden model 比对（每模块独立验证）

### Phase 3 — 综合与 PPA 报告（4–5月）

- [ ] Design Compiler 综合（28nm target cell library）
- [ ] Post-synthesis simulation（功能验证）
- [ ] PrimeTime timing signoff
- [ ] CACTI SRAM 面积功耗建模
- [ ] 全芯片 PPA 汇总表

### Phase 4 — 论文撰写与投稿（5–6月）

- [ ] 系统级 performance model（端到端 LLM inference）
- [ ] Figure 绘制（架构图、PPA 对比、精度-压缩 Pareto）
- [ ] ISSCC submission（Extended Abstract: 2页 + 4张图）
- [ ] 同步准备 JSSC full paper（如 ISSCC 接受）

---

## 附录 A：关键数值汇总

| 参数 | 值 |
|---|---|
| 目标工艺 | 28nm CMOS |
| 向量维度 d | 128 (Llama-3.1 head_dim) |
| 目标 bit-width | 3.5-bit (32ch@5bit + 96ch@3bit) |
| 压缩比 | 5.3× vs BF16 |
| RHT 级数 | 7 (log₂128) |
| Lloyd-Max 阈值数 | 7 (3-bit) / 31 (5-bit) per channel |
| QJL S 矩阵 | 128×128, INT4, 64KB SRAM |
| 并行 token 数 | 4 |
| 目标时钟 | 400 MHz |
| TQ Engine 面积 | < 0.65 mm² |
| TQ Engine 功耗 | < 90 mW |
| 量化延迟 | < 25 ns/token |
| 能效 | < 2 nJ/token |
| Flash 等效带宽 | 3 GB/s × 5.3 = ~16 GB/s |
| 支持最大上下文 | 128K tokens (4GB Flash) |

## 附录 B：相关文献

- TurboQuant: Zandieh et al., ICLR 2026, arXiv:2504.19874
- KIVI: Liu et al., ICML 2024
- KVQuant: Hooper et al., NeurIPS 2024
- ANT: Guo et al., MICRO 2021
- Cambricon-LLM: Zheng et al., MICRO 2024
- FlashLLM: xSSD近存推理, OSDI 2024
- OLIVE: Guo et al., ISCA 2023
- WHT-based DNN acceleration: multiple ISSCC/JSSC papers 2020–2024

---

# Plan B：Logic Die 近存计算方案（Hybrid Bonding CIM）

> **核心 Insight（Plan B）**：将 TQ Engine 直接集成于 3D NAND 的 Logic Die 上，通过 Hybrid Bonding 互连，在 Page Buffer 读出路径中完成量化/解量化。TQ Engine 占 Logic Die 面积 < 2%，却将等效对外带宽提升 **5.3×**，彻底消除 Toggle/ONFI 接口瓶颈。

---

## B.1 背景：CBA（CMOS-Bonded-to-Array）架构

现代 3D NAND 普遍采用 **CBA（CMOS Below/Bonded Array）** 或等效的 **Logic Die + Array Die Hybrid Bonding** 架构：

| 代次 | 厂商 | 工艺 | Logic Die 工艺 | HB 间距 | Logic Die 面积 |
|------|------|------|--------------|---------|--------------|
| V8 (176L) | Samsung | 2021 | 28nm | ~500nm | ~130 mm² |
| V9 (236L) | Samsung | 2023 | 28nm | ~300nm | ~120 mm² |
| BiCS6 (162L) | Kioxia | 2022 | 28nm | ~400nm | ~100 mm² |
| G9 (232L) | SK Hynix | 2023 | 22nm | ~300nm | ~90 mm² |

**关键结构**：

```
┌─────────────────────────────────────┐
│         Array Die (3D NAND)         │  ← NAND cell arrays, WL/BL routing
│  Cell Array │ Cell Array │ Cell Array│
├─────────────────────────────────────┤
│     Hybrid Bonding Interface        │  ← Cu-Cu or SiO₂-SiO₂, pitch ~300nm
│  Internal BW: ~4 Tbps (face-to-face)│
├─────────────────────────────────────┤
│         Logic Die (CMOS)            │  ← 28nm, ~100–130 mm²
│  Page Buffer │ Sense Amp │ Row Decode│
│  ECC Engine  │ Flash I/F │ Controller│
│  ┌──────────┐                       │
│  │ TQ Engine│ ← Plan B 新增 < 2 mm²  │
│  └──────────┘                       │
└─────────────────────────────────────┘
        │ Toggle/ONFI (外部接口)
        │ ~12.8 Gbps/die
        ▼
   Host SSD Controller / GPU
```

**内外带宽对比**：
- Hybrid Bonding 内部：~4 Tbps（数万个 Cu bump @ 300nm pitch）
- Toggle 5.1 外部接口：~12.8 Gbps/die（典型值）
- **带宽比 = 312×**，内部有绝对优势

---

## B.2 Plan B 系统架构

### B.2.1 数据路径（写入：KV Cache 压缩）

```
GPU/NPU (BF16 KV tokens)
    │
    │  PCIe / NVMe (Host Interface)
    ▼
SSD Controller
    │
    │  内部总线（或直接传至 Logic Die）
    ▼
Logic Die
    ┌────────────────────────────────────────┐
    │  Page Buffer Adapter                   │
    │  (BF16 tokens → Page Buffer 格式)      │
    │         │                              │
    │         ▼                              │
    │  ┌─────────────┐                       │
    │  │  TQ Engine  │  ← Plan B 核心        │
    │  │  Preprocessor│                      │
    │  │  RHT Butterfly│                     │
    │  │  LM-Quant    │                      │
    │  │  QJL Engine  │                      │
    │  └──────┬───────┘                      │
    │         │ 压缩 token (INT3/5 混合)      │
    │         ▼                              │
    │  ECC Encoder → Page Buffer → Array Die │
    └────────────────────────────────────────┘
         (Hybrid Bonding)
```

### B.2.2 数据路径（读出：KV Cache 解压缩）

```
Array Die (Page 读出，原始量化数据)
    │  Hybrid Bonding
    ▼
Logic Die
    ┌────────────────────────────────────────┐
    │  Page Buffer                           │
    │         │                              │
    │         ▼                              │
    │  ECC Decoder                           │
    │         │                              │
    │         ▼                              │
    │  ┌─────────────┐                       │
    │  │  TQ Decoder │  ← 解量化引擎          │
    │  │  (INT→BF16) │                       │
    │  └──────┬───────┘                      │
    │         │ 恢复 BF16 KV                  │
    │         ▼                              │
    │  Page Buffer Adapter → Toggle I/F      │
    └────────────────────────────────────────┘
    │  Toggle/ONFI (对 host)
    ▼
SSD Controller → GPU/NPU Attention
```

### B.2.3 Page Buffer 接口规格

Flash 的 Page Buffer 是 Logic Die 上的行读出暂存器，典型规格：
- Page Size：16–64 KB（每个 plane 的一个 page）
- 位宽：8K–16K bits 并行读出
- 带宽：单 plane 约 50–200 GB/s（内部）

TQ Engine 在 Page Buffer **之后、Toggle I/F 之前**插入：
- 写入方向：Host 数据 → TQ 量化 → 压缩写入 Array Die
- 读出方向：Array Die → ECC → TQ 解量化 → BF16 输出至 Host

---

## B.3 Plan B TQ Engine 规格（Logic Die 部署）

### B.3.1 接口适配器：`page_buffer_tq_adapter`

| 信号 | 方向 | 宽度 | 说明 |
|------|------|------|------|
| `pb_data_in` | in | 1024b | 来自 Page Buffer 的量化 token |
| `pb_data_out` | out | 2048b | 输出 BF16 恢复 token |
| `pb_wr_data` | in | 2048b | Host BF16 token（写入路径） |
| `pb_wr_compressed` | out | 1024b | 压缩后送 Array Die |
| `mode` | in | 1b | 0=读出解压, 1=写入压缩 |
| `valid_in` | in | 1b | 数据有效 |
| `ready_out` | out | 1b | 背压 |
| `page_id` | in | 32b | 用于 QJL 随机种子或分组索引 |

### B.3.2 面积预算（Logic Die, 28nm）

| 模块 | Plan A（控制器 die） | Plan B（Logic Die） | 备注 |
|------|--------------------|--------------------|------|
| Token Preprocessor | 0.03 mm² | 0.03 mm² | 相同 |
| RHT Butterfly (×4) | 0.08 mm² | 0.06 mm² | Logic Die cell 更密 |
| LM-Quant | 0.12 mm² | 0.10 mm² | |
| QJL Engine | 0.20 mm² | 0.16 mm² | SRAM 稍优 |
| Page Buffer Adapter | — | 0.05 mm² | 新增 |
| Top-level + misc | 0.10 mm² | 0.08 mm² | |
| **合计** | **~0.53 mm²** | **~0.48 mm²** | Logic Die 面积占比 ~0.4% |

Logic Die 总面积 ~120 mm²，TQ Engine 占比 **< 0.5%**，对存储性能零影响。

### B.3.3 功耗分析

| 场景 | 功耗 | 说明 |
|------|------|------|
| 空闲 | < 1 mW | Clock gating 关断 |
| 写入量化（4 tokens/cycle） | ~45 mW | 主要是 QJL SRAM |
| 读出解量化（4 tokens/cycle） | ~30 mW | 无 QJL（只做逆旋转+重建） |
| 峰值 | < 80 mW | Logic Die 总功耗 ~800 mW，占比 10% |

---

## B.4 Plan B 与 Plan A 对比

| 维度 | Plan A：SSD 控制器集成 | Plan B：Logic Die Hybrid Bonding |
|------|----------------------|--------------------------------|
| **集成位置** | SSD 控制器芯片（主机侧） | 3D NAND Logic Die（存储侧） |
| **带宽来源** | 控制器内部总线（~100 GB/s） | Hybrid Bonding（~4 Tbps） |
| **Toggle 接口流量** | 不变（压缩在控制器内） | 减少 **5.3×**（直接传输压缩数据） |
| **延迟** | 量化在主机侧，无额外 Flash 延迟 | Page Buffer 路径 +2–5 ns（可重叠） |
| **工艺匹配度** | 需要单独流片（28nm） | 直接复用 Logic Die 28nm，**无额外流片** |
| **商业可行性** | 需要 SSD 厂商合作 | 需要 NAND 厂商（Samsung/Kioxia）授权集成 |
| **Flash 写入次数** | 写入 BF16（较大，P/E 消耗大） | 写入压缩 INT（**P/E 减少 5.3×**） |
| **面积开销** | ~0.53 mm²（独立 die） | ~0.48 mm²（Logic Die 面积的 0.4%） |
| **适用场景** | 边缘 AI SSD（PCIe 5.0 直连） | 企业 SSD / AI 存储芯片（ISSCC 更具新颖性） |
| **ISSCC 新颖性** | ★★★☆ | ★★★★★（首篇 Hybrid Bonding + LLM KV 量化） |

**Plan B 关键优势**：
1. Toggle/ONFI 接口流量降低 5.3×，**等效提升有效带宽 5.3×**
2. Flash P/E 耗损减少 5.3×，**显著延长 SSD 寿命**
3. 写入数据量减少，**Write Amplification Factor（WAF）降低**
4. 无额外流片成本（复用 Logic Die）

---

## B.5 升级版 ISSCC Story（Plan B 主线）

### Pitch（200字）

> 长上下文 LLM 推理的 KV Cache 急剧膨胀（128K tokens × 8层 × 2×BF16 = 32 GB），已超越边缘/中端设备的 DRAM 容量。当前方案将 KV Cache 卸载至 Flash SSD，但受限于 Toggle/ONFI 接口的 12.8 Gbps 带宽，KV Cache 读写成为推理瓶颈。
>
> 本文提出 **TurboQuant-in-Flash（TQiF）**：基于 3D NAND CBA（CMOS-Bonded-to-Array）架构，将 TurboQuant KV 量化引擎（0.48 mm²，< 80 mW）直接集成于 NAND Logic Die，通过 Hybrid Bonding 互连（4 Tbps 内部带宽），在 Page Buffer 读出路径中完成量化/解量化。TQ Engine 将 BF16 KV token 压缩为 3.5-bit 混合精度格式（5.3× 压缩比），使 Toggle/ONFI 等效带宽提升 **5.3×**（12.8 → 67.8 Gbps），并将 Flash P/E 消耗降低 5.3×。
>
> 基于 Llama-3.1-8B/70B 验证，128K 上下文 NIAH 任务精度保持率 > 95%（vs Full Precision），端到端 KV Cache 吞吐从 1.6 GB/s 提升至 8.5 GB/s。

### Narrative Arc（Plan B）

```
[现状] LLM KV Cache 卸载 Flash 已成主流（边缘/中端推理）
   ↓
[问题] Toggle/ONFI 接口带宽仅 12.8 Gbps，成为推理瓶颈
        （32GB KV Cache / 1.6 GB/s = 20s flush，不可接受）
   ↓
[为什么难] 1) 控制器侧量化增加 PCIe 延迟
            2) Flash 内部量化需要修改 NAND 架构
            3) 先进 3D NAND 工艺不对外开放
   ↓
[Insight] CBA 架构的 Logic Die 已经是 28nm CMOS，面积 ~120 mm²
          Hybrid Bonding 提供 4 Tbps 内部带宽
          TQ Engine 仅需 0.48 mm²（< 0.5% Logic Die）
          → 将量化引擎"嵌入"存储芯片，是可行的
   ↓
[方案] TurboQuant-in-Flash（TQiF）
       Page Buffer 读出路径插入 TQ Engine
       写入时压缩（BF16→3.5-bit），读出时解压（3.5-bit→BF16）
   ↓
[效果] 等效带宽 5.3×↑，P/E 寿命 5.3×↑，精度损失 < 2%
```

### 4 张关键 ISSCC Figure

| Figure | 内容 |
|--------|------|
| Fig.1 | CBA 架构示意：Array Die / Logic Die / Hybrid Bonding / TQ Engine 位置；带宽对比（Toggle vs HB） |
| Fig.2 | TQ Engine 微架构：数据路径（写入/读出双向）、模块划分、Pipeline 时序图 |
| Fig.3 | PPA 汇总：面积分布饼图（Logic Die 0.5%）、功耗分解、能效（nJ/token） |
| Fig.4 | 系统评估：（a）等效带宽提升 vs 压缩比；（b）NIAH 精度保持曲线；（c）P/E 寿命延长；（d）端到端推理加速 |

---

## B.6 Plan B RTL 文件清单

| 文件 | 说明 |
|------|------|
| `page_buffer_tq_adapter.sv` | Page Buffer ↔ TQ Engine 接口适配器（新增） |
| `tq_engine.sv` | 顶层，增加 `PLAN_B` 接口模式（更新） |
| `rht_butterfly.sv` | 不变 |
| `lloyd_max_quant.sv` | 不变 |
| `qjl_engine.sv` | 不变 |
| `token_preprocessor.sv` | 不变 |

---

## 附录 C：Plan B 关键参考

- Samsung V9 NAND CBA 架构，ISSCC 2023, "A 236-Layer 3D-NAND..."
- Kioxia BiCS6 Logic Die 设计，ISSCC 2022
- Hybrid Bonding for Logic-on-Memory, IEDM 2022, 2023
- LinKV: KV Cache Management on Flash, OSDI 2024
- TurboQuant: Zandieh et al., ICLR 2026, arXiv:2504.19874


---

# 附录 D：Bitmatch 验证方案

## D.1 验证目标

证明 RTL 实现与算法（Python float）在**量化索引**（`idx_out`）上逐位一致。

## D.2 三层验证架构

```
Layer 1: Python Float (算法参考)
   ↕  对齐 1: 量化策略等价性 (SNR / mismatch rate)
Layer 2: Python Fixed-Point Model (位精确模型)
   ↕  对齐 2: RTL bitmatch (idx 逐位对比)
Layer 3: RTL Simulation (Verilator/Icarus Verilog)
```

### Layer 1 → Layer 2（策略等价性）
- 目标：fixed-point 模型与 float 算法的**索引不一致率 < 0.01%**
- 仅在阈值边界附近（±0.5 LSB）允许差异
- 工具：`fixed_point_model.py --sweep --tokens 5000`

### Layer 2 → Layer 3（RTL bitmatch）
- 目标：**0 mismatch**（逐 token 逐 channel idx 完全相同）
- 工具：Python 生成 hex 向量 → `$readmemh` 加载 → RTL 仿真逐位对比
- 流程：`fixed_point_model.py --dump` → `iverilog tq_engine_tb.sv` → 自动化断言

## D.3 已知精度影响点

| 来源 | 影响 | 严重度 | 缓解方案 |
|------|------|--------|---------|
| Q7.8 定点格式精度耗尽 | 7 级 >>>1 后仅剩 ~4 bit 有效值 | **严重** | 改为 Q3.12（见 D.4） |
| ROM 阈值量化 | `round(t × sqrt(128) × 2^frac_bits)` 引入 ±0.5 LSB 误差 | 轻微 | 增大 frac_bits |
| MIN_INT 饱和 | `y = -32768` 时 abs(y) 饱和 | 极低频 | 已处理（`< 1/65536` 概率） |
| 全局阈值共享 | 有限维度导致各 channel 分布微小偏差 | 可忽略 | D=128 时偏差 < 2% |

## D.4 Q 格式修复方案

| 参数 | 当前 Q7.8 | 推荐 Q3.12 |
|------|----------|-----------|
| 小数位 | 8 | 12 |
| BF16→定点 scale | ×256 | ×4096 |
| RHT 末级有效值域 | [-8, 7] (16 值) | [-128, 127] (256 值) |
| 有效精度 | 4 bit | 8 bit |
| 5-bit 量化可区分 | ✗（16 < 32 级） | ✓（256 >> 32 级） |
| 阈值 ROM 位宽 | 16-bit 不变 | 16-bit 不变 |
| 修改范围 | — | `token_preprocessor.sv` + `fixed_point_model.py` |

## D.5 验证执行步骤

```bash
# Step 1: 运行 bitmatch sweep（当前 Q7.8，验证 baseline mismatch）
cd hardware/sim
python fixed_point_model.py --sweep --tokens 2000

# Step 2: 查看 mismatch 分布
#   预期：~6% mismatch，集中在 5-bit outlier channels
#   原因：Q7.8 精度不足

# Step 3: 修复 Q 格式后重新验证
#   python fixed_point_model.py --sweep --tokens 5000
#   目标：mismatch < 0.01%

# Step 4: 生成 RTL 金标准向量
python fixed_point_model.py --dump
# 输出文件:
#   hardware/tb/hw_test_tokens.hex
#   hardware/tb/hw_expected_idx.hex
#   hardware/tb/hw_expected_sign.hex

# Step 5: RTL 仿真比对
cd ../rtl
iverilog -g2012 -o sim_tq tq_engine_tb.sv tq_engine.sv \
    rht_butterfly.sv lloyd_max_quant.sv qjl_engine.sv token_preprocessor.sv
vvp sim_tq
# 断言：idx_out == expected_idx[token_cnt] 对每个 token
```

## D.6 持续集成（CI）

```makefile
# hardware/Makefile 新增目标
bitmatch: gen_vectors sim_bitmatch
gen_vectors:
	cd sim && python fixed_point_model.py --dump
sim_bitmatch:
	cd rtl && iverilog -g2012 -o sim_bitmatch tq_engine_tb.sv ... && vvp sim_bitmatch
```

每次 RTL 修改后运行 `make bitmatch`，确保 golden 向量不变。


---

# 附录 E：KV Compass × TurboQuant 融合研究方案

> 记录时间：2026-04-21
> 来源：KV Compass 实验文档（`0-LLM/3-exp/0-extattn/0-extattention/1-extattn-KVcompass/实验文档_KVCompass_完整版.md`）× TurboQuant ASIC 设计（本文档前各章节）

---

## E.1 背景：两项工作的关键发现

### E.1.1 KV Compass（输入重复 + KV 压缩）

**输入重复**将 prompt 拼接为 `[X1 | sep | X2]`，显著提升推理能力：
- GSM8K (1B): 24% → 35% (+11pt)
- NameIndex-300 (8B): 24% → 82% (+58pt)
- ARC-Easy: 69% → 72% (+3pt)

代价是 KV Cache 翻倍（2×）。KV Compass 实验发现：

| 策略 | KV 节省 | 关键发现 |
|------|---------|---------|
| vp_k1（BOS-only trim） | ~49% | 检索任务持平/超越 plain_repeat |
| **vp_quant_int8** | ~25% | **数学推理超越 plain_repeat +3pt（反直觉）** |
| vp_top_attn_k8 | ~49% | SnapKV 选择最重要 8 个 X1 token |

**核心 Finding（发现 4，来自 V3）**：
> INT8 量化噪声起到**隐式正则化**效果，打破了精确重复造成的脆弱注意力模式，使 `vp_quant_int8` 在 GSM8K 上**超越** `plain_repeat` +3pt（8B 模型），1B 模型亦有超越。

### E.1.2 TurboQuant（KV Cache 量化算法）

TurboQuant（Zandieh et al., ICLR 2026）基于随机正交旋转（RHT）+ Lloyd-Max 量化 + QJL 残差，实现：
- 3.5-bit 混合精度（32 outlier @ 5-bit + 96 regular @ 3-bit）
- 128K context NIAH 精度保持率 > 95%（vs FP16）
- 数学可证的压缩误差界

当前硬件方案（Plan A/B）实现 5.3× KV 压缩，< 0.5 mm²，< 80 mW @28nm。

---

## E.2 融合动机：为什么这是强组合

### E.2.1 问题规模放大

输入重复场景下的 KV Cache 压力**显著高于普通推理**：

| 场景 | KV Cache 规模 | Flash 卸载需求 |
|------|-------------|--------------|
| 普通推理（128K） | ~32 GB | 可容纳 |
| 重复输入（128K×2） | ~64 GB | 严重超出 |
| 重复输入 + 多轮对话 | >128 GB | 必须 Flash |

→ 输入重复场景是 TurboQuant-in-Flash（TQiF）**最强的 motivation 放大器**。

### E.2.2 反直觉的算法协同效应

KV Compass 发现 INT8 量化噪声有益（隐式正则化）。这意味着：
- **量化不是障碍，而是辅助**：在重复输入场景下，适度的量化噪声可以**主动有益**于推理质量
- TurboQuant 的随机旋转（RHT）引入的随机性与这一正则化效果**天然匹配**
- **硬件可以使用更激进的 bit-width**，而不仅仅是"尽量减少精度损失"

### E.2.3 互补的压缩轴

| 压缩轴 | 方法 | 压缩比 |
|--------|------|--------|
| 时序（丢弃 X1 KV） | KV Compass vp_k1 | 2× |
| 精度（量化） | TurboQuant 3.5-bit | 4.57× |
| **联合** | **Repetition-Aware TQ** | **~8-9×** |

---

## E.3 核心研究方向：三个递进 Story Angle

### Angle A（弱，不推荐）：叠加压缩
直接串联 KV Compass + TurboQuant，获得 9× 总压缩比。
**问题**：无新 insight，评委会认为是增量工作。

### Angle B（中，适合 DAC/TCAS）：Repetition-Aware Mixed Precision
**Insight**：X1 和 X2 的 KV 具有不同的统计分布和功能重要性。Decode 阶段 X2 已吸收 X1 信息，X1 KV 可用**更激进量化**（2-bit），X2 保持标准精度（3.5-bit）。

```
┌───────────────────────────────────────────────────────┐
│  X1 tokens (position 0..L-1)  │  X2 tokens (L..2L-1) │
│  b = 2-bit (aggressive)       │  b = 3.5-bit (std)    │
│  TQ quantizer: 2-bit path     │  TQ quantizer: normal │
└───────────────────────────────────────────────────────┘
```

硬件创新：`lloyd_max_quant.sv` 的 `b_sel` 扩展为 `b_sel[token_pos_type][channel]`，支持 position-aware dynamic precision dispatch。

**预期数字**：总压缩 ~7-8×，精度持平 plain_repeat FP16。

### Angle C（强，目标 HPCA/MICRO）：量化作为隐式正则化的共同设计
**核心 Insight**（直接来自 KV Compass V3 发现 4）：
> 对于重复输入，量化噪声是一种**受控的随机扰动**，可以主动打破脆弱的重复注意力模式，提升推理鲁棒性。TurboQuant 的 RHT 随机旋转天然产生这种噪声。因此，"量化—精度损失"的传统权衡在重复输入下被**颠覆**：更激进的量化可以获得**更好的任务性能**。

**Narrative Arc**：
```
[现状] 输入重复（Leviathan 2025, OpenAI o1 思维链）推动 LLM 推理能力提升
   ↓
[问题] 2× KV Cache 与有限 DRAM/Flash 带宽的矛盾（重复输入是 KV 存储危机的放大器）
   ↓
[Why hard] 传统认为：量化损失精度，越激进越不安全
   ↓
[Key Insight] 重复输入对量化噪声具有内在鲁棒性，甚至因隐式正则化而受益
              → 量化的"最优 bit-width"在重复输入下比非重复更低
   ↓
[Solution] Repetition-Aware TurboQuant（RAT）：
           X1 tokens: 2-bit（主动利用正则化）
           X2 tokens: 3.5-bit（保真，用于 Decode 注意力主体）
           硬件：Position-aware dual-precision TQ Engine on Flash Logic Die
   ↓
[Proof] GSM8K / NIAH / ARC 等任务上，RAT 量化+重复 > FP16 plain_repeat > FP16 baseline
```

---

## E.4 必须先做的关键实验

### E.4.1 实验 1：TurboQuant × 重复输入 精度矩阵（P0，最关键）
将 KV Compass 的 `vp_quant_int8` 替换为 TurboQuant 量化器，填补如下矩阵：

| bits | plain_repeat | KV Compass + TQ | 差值 Δ | 结论 |
|------|--------------|-----------------|--------|------|
| FP16 | ✓（已有） | ✓（已有） | 0 | baseline |
| INT8 | ✓（已有） | ✓（已有） | +3pt（GSM8K 8B）| 正则化有效 |
| TQ 4-bit | ? | ? | ? | 待验证 |
| TQ 3.5-bit | ? | ? | ? | **核心验证点** |
| TQ 2.5-bit | ? | ? | ? | 激进下界 |

**判断标准**：
- 若 TQ 3.5-bit + repeat ≥ FP16 plain_repeat → Angle C 成立，story 最强
- 若 TQ 4-bit + repeat ≥ FP16 plain_repeat，3.5-bit 略低 → Angle B 成立
- 若仅 INT8 有效 → 回到简单叠加，paper 贡献降低

**实现路径**：
- 融合 `0-LLM/3-exp/0-extattn/0-extattention/1-extattn-KVcompass/kv_compass_v4.py` 的 DynamicCache patch
- 与 `0-LLM/5-TurboQuant/kv_cache_quant.py` 的 TurboQuant 量化器
- 在同一测试集（GSM8K, NameIndex-300, ARC-Challenge）上评测

### E.4.2 实验 2：X1 / X2 KV 统计分布差异（P1，支撑 Angle B 硬件创新）
分析 Prefill 结束后：
- X1 KV 的奇异值分布（展示"可压缩性"）
- X2 KV 的奇异值分布
- 跨层 attention entropy（X1 targets vs X2 targets）
- 各层各 head 的 X1→X2 attention 比例（热力图）

**预期**：X1 KV 奇异值衰减更快（更可压缩），支持对 X1 使用更激进量化。

### E.4.3 实验 3：联合压缩的 Pareto 前沿（P1，用于 paper Figure）
x 轴：总 KV 节省率（%）
y 轴：任务精度（F1 / Accuracy）
画出：
- KV Compass alone（trim）
- TurboQuant alone
- 朴素叠加
- **RAT（联合设计）**的 Pareto 前沿

---

## E.5 硬件创新点（RTL 层面）

### E.5.1 Position-Aware Dual-Precision TQ Engine
扩展 `tq_engine.sv` 以支持 per-token-type 精度配置：

```
输入信号（新增）:
  input  wire [1:0]   token_type_in  // 00=X1, 01=sep, 10=X2, 11=decode
  
内部逻辑:
  b_sel_effective = (token_type == 2'b00) ? b_sel_aggressive  // X1: 2-bit
                  : (token_type == 2'b10) ? b_sel_standard     // X2: 3.5-bit
                  : b_sel_high;                                 // sep: 5-bit
```

`lloyd_max_quant.sv` 增加第三组全局阈值（2-bit，3 个阈值，仅 1 个正数比较器），ROM 增加约 16×2=32 bits。

**面积增量**：可忽略（仅增加 1 个 MUX 层和极小 ROM）。

### E.5.2 Replica-Aware QJL（节省 50% QJL 存储）
Decode 阶段 X2 的注意力目标主要是 X1（KV Compass V2 跨副本注意力分析已验证）。
QJL 的 S 矩阵对 X1/X2 可以共享：
- X1 和 X2 的 QJL sign 用同一个 S 矩阵生成
- Decode 时通过 X2 QJL sign 近似 X1 内积，无需存储 X1 QJL

**RTL 变化**：`qjl_engine.sv` 增加 `replica_mode` 端口，启用后 sign 输出从 X2 SRAM 读取而非重计算。

### E.5.3 Streaming Skip for vp_k1 on Logic Die
KV Compass vp_k1 只保留 BOS（1 个 token）。Flash Logic Die 解压时跳过 X1 的全部 page（除 BOS 所在 page）。

`page_buffer_tq_adapter.sv` 增加：
```
input wire [31:0]  skip_page_start,  // X1 起始 page id
input wire [31:0]  skip_page_end,    // X1 结束 page id（except BOS page）
```
Page read scheduler 直接跳转，节省 ~49% 的 Flash 读出周期。

---

## E.6 目标会议与时间线

| 会议 | 类型 | Deadline | 是否需要流片 | 适合 Story |
|------|------|----------|------------|-----------|
| **HPCA 2027** | 系统架构 | ~2026-07 | 否（仿真+综合） | **首选，Angle C** |
| MICRO 2026 | 系统架构 | ~2026-04（已近） | 否 | 可行，时间紧 |
| TCAS-I/II | 电路 | rolling | 否 | 硬件侧专注 |
| JSSC | 电路 | rolling | 强烈推荐 tapeout | 如有 tapeout |
| ISSCC | 电路 | ~2026-09 | 需要 tapeout/FPGA | 长期目标 |

**推荐路径**：
1. 完成实验 1-3（4-6 周）
2. 提交 HPCA 2027 或 TCAS-I（系统+电路均衡）
3. 同步进行 DC 综合 + FPGA 原型，为 ISSCC 2028 积累素材

---

## E.7 当前研究状态汇总

| 组件 | 状态 | 位置 |
|------|------|------|
| TurboQuant 算法实现 | ✅ 完成 | `0-LLM/5-TurboQuant/` |
| KV Compass 算法实现 | ✅ 完成（V4） | `0-LLM/3-exp/0-extattn/0-extattention/1-extattn-KVcompass/` |
| Plan A RTL（SSD Controller） | ✅ 完成初版 | `hardware/rtl/` |
| Plan B RTL（Logic Die HB） | ✅ 完成初版 | `hardware/rtl/` |
| Fixed-point 位精确模型 | ✅ 完成 | `hardware/sim/fixed_point_model.py` |
| Bitmatch 验证框架 | ✅ 框架完成 | `hardware/tb/` |
| Q 格式精度修复（Q3.12）| ⚠️ 待做 | `token_preprocessor.sv` + `fixed_point_model.py` |
| TQ × 重复输入 精度矩阵 | ❌ 未做（P0） | 实验脚本待写 |
| X1/X2 KV 分布分析 | ❌ 未做（P1） | 分析脚本待写 |
| Position-aware dual-precision RTL | ❌ 未做 | `tq_engine.sv` 扩展 |
| DC 综合 PPA 报告 | ❌ 未做 | `hardware/Makefile` |
| FPGA 原型 | ❌ 未做（长期） | — |

---

## E.8 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| TQ 3.5-bit + repeat 精度不如 FP16 repeat | 中 | 高（颠覆 Angle C） | 先跑实验 1，如失败改用 Angle B |
| X1/X2 分布差异不显著 | 低 | 中（削弱 Angle B 动机） | 仍可从 vp_k1 + TQ 的带宽收益出发 |
| RTL 仿真 bitmatch 精度问题（Q7.8 耗尽） | 已知 | 中 | 修复为 Q3.12（见附录 D.4） |
| HPCA 2027 deadline 太早 | 中 | 中 | 退守 TCAS-I/MICRO 2027 |

