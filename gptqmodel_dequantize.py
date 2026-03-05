"""
从 GPTQModel 量化的模型中提取伪量化（dequantized）的模型

这允许你在不同的 transformers 版本环境中使用量化后的模型，
而无需任何性能加速，完全兼容旧版 transformers。

使用场景：
- GPTQModel 依赖 transformers 4.57.1
- openvla 依赖 transformers 4.40.0
- 需要在两个环境中都使用同一个模型

解决方案：
将量化模型反量化（dequantize）为 FP16/BF16 的标准模型
这样就能在任何 transformers 版本中加载使用
"""

import torch
from pathlib import Path
from gptqmodel.utils.model_dequant import dequantize_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dequantize_gptq_model(
    quantized_model_path: str,
    output_model_path: str,
    target_dtype: str = "bfloat16",
    device: str = None
) -> None:
    """
    将 GPTQModel 量化的模型反量化为标准格式

    参数：
        quantized_model_path: 量化模型的路径（GPTQModel 保存的路径）
        output_model_path: 输出路径（未量化的模型将保存在这里）
        target_dtype: 目标数据类型 ("bfloat16" 或 "float16")
        device: 使用的设备 ("cuda:0" 或 "cpu")，None 表示自动选择

    例子：
        # 从 openvla-7b-gptqmodel-libero 提取未量化模型
        dequantize_gptq_model(
            quantized_model_path="openvla-7b-gptqmodel-libero",
            output_model_path="openvla-7b-dequantized",
            target_dtype="bfloat16",
            device="cuda:0"
        )
    """

    # 转换参数
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if target_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {target_dtype}. Choose from {list(dtype_map.keys())}")

    target_dtype_torch = dtype_map[target_dtype]

    # 验证输入路径
    quantized_path = Path(quantized_model_path)
    if not quantized_path.exists():
        raise FileNotFoundError(f"Quantized model path not found: {quantized_path}")

    # 验证输出路径
    output_path = Path(output_model_path)
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    logger.info("=" * 80)
    logger.info("GPTQModel 反量化工具")
    logger.info("=" * 80)
    logger.info(f"输入（量化模型）: {quantized_path.absolute()}")
    logger.info(f"输出（未量化模型）: {output_path.absolute()}")
    logger.info(f"目标数据类型: {target_dtype}")
    logger.info(f"使用设备: {device or '自动选择'}")
    logger.info("=" * 80)

    try:
        # 执行反量化
        logger.info("\n开始反量化过程...")
        dequantize_model(
            model_path=quantized_path,
            output_path=output_path,
            target_dtype=target_dtype_torch,
            device=device
        )

        logger.info("✓ 反量化完成！")
        logger.info(f"\n已保存到: {output_path.absolute()}")
        logger.info("\n现在你可以使用以下方式加载模型：")
        logger.info(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{output_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    "{output_model_path}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
        """)

    except Exception as e:
        logger.error(f"✗ 反量化失败: {e}")
        logger.error("\n常见问题:")
        logger.error("1. 确保输入路径包含 .safetensors 文件")
        logger.error("2. 确保 config.json 存在")
        logger.error("3. 如果使用 GPU，确保 CUDA 可用且显存充足")
        logger.error("4. 如果显存不足，使用 device='cpu' 在 CPU 上反量化（会很慢）")
        raise


def compare_model_info(model_path: str) -> None:
    """
    显示模型的信息（量化配置等）
    """
    import json
    from pathlib import Path

    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info("\n模型配置信息:")
    logger.info("=" * 60)

    # 基本信息
    logger.info(f"Model type: {config.get('model_type', 'Unknown')}")
    logger.info(f"Architecture: {config.get('architectures', ['Unknown'])[0]}")

    # 量化配置
    quant_config = config.get('quantization_config')
    if quant_config:
        logger.info(f"\n量化配置:")
        logger.info(f"  - 方法: {quant_config.get('quant_method', 'Unknown')}")
        logger.info(f"  - 位数: {quant_config.get('bits', 'Unknown')}")
        logger.info(f"  - Group size: {quant_config.get('group_size', 'Unknown')}")
        logger.info(f"  - Format: {quant_config.get('fmt', 'Unknown')}")
    else:
        logger.info("\n该模型没有量化配置（已是未量化模型）")

    logger.info("=" * 60)


# ============================================================================
# 使用示例和完整工作流
# ============================================================================

def complete_workflow_example():
    """
    完整的工作流示例：
    1. 查看量化模型信息
    2. 反量化模型
    3. 使用反量化后的模型
    """

    print("\n" + "=" * 80)
    print("完整工作流示例")
    print("=" * 80)

    # 配置
    quantized_model = "openvla-7b-gptqmodel-libero"  # GPTQModel 保存的量化模型
    output_model = "openvla-7b-dequantized"  # 输出的反量化模型

    print("\n步骤 1: 查看量化模型信息")
    print("-" * 80)
    compare_model_info(quantized_model)

    print("\n步骤 2: 反量化模型")
    print("-" * 80)
    print("这将把量化权重转换回 BF16 格式...")
    print(f"输入: {quantized_model}")
    print(f"输出: {output_model}")
    print("\n选择执行方式：")
    print("- GPU 上反量化（推荐，快速）: dequantize_gptq_model(..., device='cuda:0')")
    print("- CPU 上反量化（慢，但不需要 GPU）: dequantize_gptq_model(..., device='cpu')")

    # 实际执行（如果路径存在的话）
    # dequantize_gptq_model(
    #     quantized_model_path=quantized_model,
    #     output_model_path=output_model,
    #     target_dtype="bfloat16",
    #     device="cuda:0"  # 或 "cpu"
    # )

    print("\n步骤 3: 使用反量化后的模型")
    print("-" * 80)
    print(f"""
# 现在可以在任何 transformers 版本中使用（包括 4.40.0）
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("{output_model}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "{output_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 推理
inputs = tokenizer("The robot should", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
    """)


# ============================================================================
# 快速参考
# ============================================================================

def quick_reference():
    """打印快速参考指南"""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    GPTQModel 反量化快速参考指南                             ║
╚════════════════════════════════════════════════════════════════════════════╝

问题：
  GPTQModel 依赖 transformers 4.57.1
  openvla 依赖 transformers 4.40.0
  → 无法在两个环境中同时使用量化模型

解决方案：
  将量化模型反量化（dequantize）为标准 FP16/BF16 格式
  → 兼容所有 transformers 版本

────────────────────────────────────────────────────────────────────────────

使用步骤：

1️⃣ 在 GPTQModel 环境中执行反量化：

    from gptqmodel_dequantize import dequantize_gptq_model
    
    dequantize_gptq_model(
        quantized_model_path="openvla-7b-gptqmodel-libero",
        output_model_path="openvla-7b-dequantized",
        target_dtype="bfloat16",
        device="cuda:0"  # 推荐使用 GPU
    )

2️⃣ 在 openvla 环境中加载反量化后的模型：

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained("openvla-7b-dequantized", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openvla-7b-dequantized",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

────────────────────────────────────────────────────────────────────────────

参数说明：

  quantized_model_path:  量化模型的路径（GPTQModel 保存的位置）
  output_model_path:     输出路径（新的反量化模型将保存在这里）
  target_dtype:          目标数据类型
                         - "bfloat16"（推荐，精度高，省显存）
                         - "float16"（标准，精度较好）
                         - "float32"（最高精度，但显存消耗多）
  device:                使用的设备
                         - "cuda:0"（推荐，快速）
                         - "cuda:1"、"cuda:2" 等（如果有多 GPU）
                         - "cpu"（显存不足时使用，会很慢）
                         - None（自动选择）

────────────────────────────────────────────────────────────────────────────

优势和权衡：

✓ 优势：
  - 兼容所有 transformers 版本
  - 无需任何性能加速的复杂代码
  - 模型结构和权重完全标准化
  - 可以使用任何标准推理框架（vLLM、TensorRT 等）

⚠ 权衡：
  - 推理速度不如量化版本快（但仍比原始 BF16 快不了多少）
  - 显存占用是量化版本的 4 倍左右（但仍比 FP32 少）
  - 反量化过程需要 1-5 分钟（取决于模型大小和硬件）

────────────────────────────────────────────────────────────────────────────

常见问题：

Q: 反量化会失去量化带来的性能提升吗？
A: 是的。如果你需要性能，应该在同一个 transformers 版本中使用量化模型。
   反量化是为了解决版本不兼容的问题。

Q: 反量化的模型和原始模型有什么区别？
A: 在数学上是等价的。量化 → 反量化 ≈ 原始模型（可能有极小的精度差异）

Q: 如何在保留性能的情况下解决版本冲突？
A: 升级 openvla 依赖到更新的 transformers 版本（如果可能的话）。
   这样就能直接使用量化模型并获得性能优势。

Q: 反量化文件大小会增加多少？
A: BF16 反量化：显存占用大约是量化版本的 4 倍
   float16 反量化：同上
   float32 反量化：大约是量化版本的 8 倍

Q: 反量化需要多长时间？
A: 取决于模型大小：
   - 7B 模型：1-2 分钟（GPU）
   - 13B 模型：2-3 分钟（GPU）
   - 70B 模型：5-10 分钟（GPU）
   在 CPU 上会慢 10-100 倍。

────────────────────────────────────────────────────────────────────────────

性能对比（7B 模型）：

方式                          推理速度    显存占用      兼容性
─────────────────────────────────────────────────────────────
量化模型（GPTQ 4-bit）        ⭐⭐⭐⭐⭐     ~3.5GB        需要 4.57.1
反量化 BF16                   ⭐⭐⭐⭐      ~12GB         所有版本 ✓
原始 BF16 模型                ⭐⭐⭐⭐      ~12GB         所有版本 ✓
FP32 模型                     ⭐⭐⭐       ~28GB         所有版本 ✓

────────────────────────────────────────────────────────────────────────────
    """)


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="从 GPTQModel 量化模型中提取伪量化（反量化）模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用 GPU 反量化
  python gptqmodel_dequantize.py \\
    --quantized_model openvla-7b-gptqmodel-libero \\
    --output_model openvla-7b-dequantized \\
    --dtype bfloat16 \\
    --device cuda:0
  
  # 使用 CPU 反量化（不需要 GPU）
  python gptqmodel_dequantize.py \\
    --quantized_model openvla-7b-gptqmodel-libero \\
    --output_model openvla-7b-dequantized \\
    --dtype bfloat16 \\
    --device cpu
  
  # 查看快速参考
  python gptqmodel_dequantize.py --quick-ref
  
  # 查看模型信息
  python gptqmodel_dequantize.py --info openvla-7b-gptqmodel-libero
        """
    )

    parser.add_argument(
        "--quantized_model",
        default='/home/Qitao/project/GPTQModel/openvla-7b-gptqmodel-libero',
        type=str,
        help="量化模型的路径"
    )
    parser.add_argument(
        "--output_model",
        default='/home/Qitao/project/GPTQModel/openvla-7b-dequantized',
        type=str,
        help="输出模型的路径"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="目标数据类型（默认：bfloat16）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用的设备（默认：自动选择）。例如：cuda:0, cpu"
    )
    parser.add_argument(
        "--quick-ref",
        action="store_true",
        help="显示快速参考指南"
    )
    parser.add_argument(
        "--info",
        type=str,
        help="查看模型的量化配置信息"
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="显示完整工作流示例"
    )

    args = parser.parse_args()

    # 处理不同的命令
    if args.quick_ref:
        quick_reference()

    elif args.info:
        compare_model_info(args.info)

    elif args.workflow:
        complete_workflow_example()

    elif args.quantized_model and args.output_model:
        dequantize_gptq_model(
            quantized_model_path=args.quantized_model,
            output_model_path=args.output_model,
            target_dtype=args.dtype,
            device=args.device
        )

    else:
        parser.print_help()
        print("\n提示：运行 'python gptqmodel_dequantize.py --quick-ref' 查看快速参考")


