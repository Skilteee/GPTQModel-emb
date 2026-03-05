import argparse
import struct

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from gptqmodel import GPTQModel
from huggingface_hub import hf_hub_download


def load_calib_bin(path):
    with open(path, "rb") as f:
        magic = f.read(16)
        version = struct.unpack("<I", f.read(4))[0]
        num_samples = struct.unpack("<I", f.read(4))[0]
        hidden_dim = struct.unpack("<I", f.read(4))[0]
        _ = struct.unpack("<IIII", f.read(16))
        _ = struct.unpack("<I", f.read(4))

        seq_lens = np.frombuffer(f.read(num_samples * 4), dtype=np.uint32)
        offsets = np.frombuffer(f.read(num_samples * 8), dtype=np.uint64)

        data_start = f.tell()
        embeddings = []
        for i in range(num_samples):
            f.seek(data_start + int(offsets[i]))
            n_elem = int(seq_lens[i]) * hidden_dim
            emb = np.frombuffer(f.read(n_elem * 4), dtype=np.float32).reshape(
                int(seq_lens[i]),
                hidden_dim,
            )
            embeddings.append(emb)

    return {
        "magic": magic,
        "version": version,
        "hidden_dim": hidden_dim,
        "sequence_lengths": seq_lens,
        "embeddings": embeddings,
    }


def _mean_cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


def _mse(a, b):
    return torch.mean((a.float() - b.float()) ** 2).item()


def _mae(a, b):
    return torch.mean((a.float() - b.float()).abs()).item()


def compare_models(fp_model, gptq_model, embeddings, max_samples=None):
    cos_hidden, mse_hidden, mae_hidden = [], [], []
    cos_logits, mse_logits, mae_logits = [], [], []

    for idx, emb in enumerate(embeddings):
        if max_samples is not None and idx >= max_samples:
            break

        inputs_embeds = torch.tensor(emb, device=fp_model.device, dtype=fp_model.dtype).unsqueeze(0)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
        )

        with torch.no_grad():
            fp_out = fp_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            gptq_out = gptq_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        fp_hidden = fp_out.hidden_states[-1]
        gptq_hidden = gptq_out.hidden_states[-1]
        cos_hidden.append(_mean_cos(fp_hidden, gptq_hidden))
        mse_hidden.append(_mse(fp_hidden, gptq_hidden))
        mae_hidden.append(_mae(fp_hidden, gptq_hidden))

        # import code
        # code.interact(local=locals())

        fp_logits = fp_out.logits[:, -1, :]
        gptq_logits = gptq_out.logits[:, -1, :]
        cos_logits.append(_mean_cos(fp_logits, gptq_logits))
        mse_logits.append(_mse(fp_logits, gptq_logits))
        mae_logits.append(_mae(fp_logits, gptq_logits))

    metrics = {
        "hidden_cos_mean": float(np.mean(cos_hidden)),
        "hidden_mse_mean": float(np.mean(mse_hidden)),
        "hidden_mae_mean": float(np.mean(mae_hidden)),
        "logits_cos_mean": float(np.mean(cos_logits)),
        "logits_mse_mean": float(np.mean(mse_logits)),
        "logits_mae_mean": float(np.mean(mae_logits)),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_model", default='/home/Qitao/project/GPTQModel/openvla7b_llm', help="Path to original FP/BF16 LLM.")
    parser.add_argument("--gptq_model", default='/home/Qitao/project/GPTQModel/openvla-7b-gptqmodel-libero', help="Path to GPTQ quantized LLM.")
    parser.add_argument("--calib_bin", help="Path to calibration .bin file.")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to compare.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    fp_model = AutoModelForCausalLM.from_pretrained(
        args.fp_model,
        # "/home/Qitao/project/GPTQModel/openvla-7b-dequantized",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    fp_model.eval()

    # gptq_model = AutoModelForCausalLM.from_pretrained(
    #     '/home/Qitao/project/GPTQModel/openvla-7b-dequantized',
    #     torch_dtype=dtype,
    #     low_cpu_mem_usage=True,
    # ).to(args.device)

    gptq_model = GPTQModel.load(args.gptq_model, device=args.device)
    gptq_model.eval()

    bin_path = hf_hub_download(
        repo_id='arashakb/llm-embedding-calib-data-libero',
        filename='spatial_calib_10_episodes.bin',
        # repo_type="dataset"  # 关键：这是 dataset，不是 model
    )

    calib = load_calib_bin(bin_path)
    metrics = compare_models(
        fp_model,
        gptq_model,
        calib["embeddings"],
        max_samples=args.max_samples,
    )

    print("Comparison metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6g}")


if __name__ == "__main__":
    main()
