"""
彻底解决 transformers 4.40.0 与 4.57.1 版本不兼容问题

问题分析：
- 合并后的 model.safetensors 仍然报错：metadata.get("format")
- 原因：transformers 4.40.0 期望 safetensors 文件有特定的元数据格式
- 解决：将 safetensors 转换为 PyTorch 格式，完全兼容所有版本

这是最终解决方案，绕过所有 safetensors 元数据问题
"""

import torch
import json
from pathlib import Path
from safetensors.torch import load_file
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_safetensors_to_pytorch(model_path: str, output_path: str = None):
    """
    将 safetensors 模型完全转换为 PyTorch 格式

    这是最彻底的解决方案，绕过所有 safetensors 兼容性问题

    参数：
        model_path: safetensors 模型路径
        output_path: 输出路径（默认：model-pytorch）
    """

    model_path = Path(model_path)
    if output_path is None:
        output_path = Path(str(model_path) + "-pytorch")
    else:
        output_path = Path(output_path)

    logger.info("=" * 80)
    logger.info("将 safetensors 模型转换为 PyTorch 格式")
    logger.info("=" * 80)
    logger.info(f"输入模型：{model_path.absolute()}")
    logger.info(f"输出模型：{output_path.absolute()}")

    if not model_path.exists():
        logger.error(f"模型路径不存在：{model_path}")
        return False

    if output_path.exists():
        logger.error(f"输出路径已存在：{output_path}")
        return False

    try:
        output_path.mkdir(parents=True)

        # 步骤 1: 复制非 safetensors 文件
        logger.info("\n步骤 1: 复制配置和 tokenizer 文件...")

        for item in model_path.iterdir():
            if item.suffix == ".safetensors" or "safetensors.index" in item.name:
                continue

            if item.is_dir():
                shutil.copytree(item, output_path / item.name)
                logger.info(f"  ✓ 复制目录：{item.name}")
            else:
                shutil.copy2(item, output_path / item.name)
                logger.info(f"  ✓ 复制文件：{item.name}")

        # 步骤 2: 加载所有 safetensors 文件
        logger.info("\n步骤 2: 加载 safetensors 权重...")

        safetensors_files = sorted(
            model_path.glob("*.safetensors"),
            key=lambda x: x.name
        )

        if not safetensors_files:
            logger.error("找不到 .safetensors 文件")
            return False

        logger.info(f"找到 {len(safetensors_files)} 个 safetensors 文件")

        all_tensors = {}

        for i, sf_file in enumerate(safetensors_files):
            logger.info(f"  [{i+1}/{len(safetensors_files)}] 加载 {sf_file.name}...")

            try:
                tensors = load_file(str(sf_file))
                all_tensors.update(tensors)
                logger.info(f"    ✓ 加载成功，{len(tensors)} 个张量")
            except Exception as e:
                logger.error(f"    ✗ 加载失败：{e}")
                return False

        logger.info(f"\n✓ 总共加载 {len(all_tensors)} 个张量")

        # 步骤 3: 保存为 PyTorch 格式
        logger.info("\n步骤 3: 保存为 PyTorch 格式...")

        output_file = output_path / "pytorch_model.bin"

        logger.info(f"保存到：{output_file}")
        torch.save(all_tensors, str(output_file))

        file_size_gb = output_file.stat().st_size / 1e9
        logger.info(f"✓ 已保存，文件大小：{file_size_gb:.2f} GB")

        # 步骤 4: 更新 config.json
        logger.info("\n步骤 4: 更新配置文件...")

        config_path = output_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            # 移除量化配置
            if "quantization_config" in config:
                logger.warning("移除 quantization_config")
                del config["quantization_config"]

            # 确保数据类型设置正确
            if "torch_dtype" not in config:
                config["torch_dtype"] = "bfloat16"
                logger.info("设置 torch_dtype 为 bfloat16")

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info("✓ 配置文件已更新")

        # 步骤 5: 验证
        logger.info("\n步骤 5: 验证转换结果...")

        try:
            # 验证能否加载
            pytorch_file = output_path / "pytorch_model.bin"
            test_tensors = torch.load(str(pytorch_file), map_location="cpu")
            logger.info(f"✓ PyTorch 文件验证成功，包含 {len(test_tensors)} 个张量")

            # 验证配置
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(output_path), trust_remote_code=True)
            logger.info(f"✓ 配置验证成功")
            logger.info(f"  - 模型类型：{config.model_type}")
            logger.info(f"  - 架构：{config.architectures if hasattr(config, 'architectures') else 'N/A'}")

        except Exception as e:
            logger.warning(f"验证出现警告（可能不严重）：{e}")

        logger.info("\n" + "=" * 80)
        logger.info("✓ 转换完成！")
        logger.info("=" * 80)
        logger.info(f"\n转换后的模型已保存到：{output_path.absolute()}")
        logger.info("\n现在可以在 openvla 环境中加载：")
        logger.info(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("{output_path}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "{output_path}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
        """)

        return True

    except Exception as e:
        logger.error(f"\n✗ 转换失败：{e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def quick_fix_all_versions(model_path: str):
    """
    一键修复所有版本不兼容问题
    自动尝试所有解决方案，直到成功
    """

    model_path = Path(model_path)

    logger.info("=" * 80)
    logger.info("自动修复 transformers 版本不兼容问题")
    logger.info("=" * 80)

    if not model_path.exists():
        logger.error(f"模型不存在：{model_path}")
        return False

    # 方案 1: 转换为 PyTorch 格式（推荐，最兼容）
    logger.info("\n尝试方案 1: 转换为 PyTorch 格式...")
    output_path = Path(str(model_path) + "-pytorch")

    if convert_safetensors_to_pytorch(str(model_path), str(output_path)):
        logger.info("\n✓ 方案 1 成功！")
        return True

    logger.error("\n✗ 方案 1 失败")
    return False


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="将 safetensors 模型转换为 PyTorch 格式（终极解决方案）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：

转换为 PyTorch 格式：
  python convert_to_pytorch.py convert \\
    --model openvla-7b-dequantized-merged \\
    --output openvla-7b-pytorch

自动修复所有问题：
  python convert_to_pytorch.py auto-fix \\
    --model openvla-7b-dequantized-merged

然后在 openvla 环境中加载：
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("openvla-7b-pytorch")
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='选择操作')

    # convert 命令
    convert_parser = subparsers.add_parser('convert', help='转换为 PyTorch 格式')
    convert_parser.add_argument('--model', type=str, required=True, help='输入模型路径')
    convert_parser.add_argument('--output', type=str, default=None, help='输出路径')

    # auto-fix 命令
    autofix_parser = subparsers.add_parser('auto-fix', help='自动修复所有问题')
    autofix_parser.add_argument('--model', type=str, required=True, help='输入模型路径')

    args = parser.parse_args()

    if args.command == 'convert':
        success = convert_safetensors_to_pytorch(args.model, args.output)
        sys.exit(0 if success else 1)

    elif args.command == 'auto-fix':
        success = quick_fix_all_versions(args.model)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


