import torch
from datasets import load_dataset
from transformers import AutoTokenizer
# from transformers import AutoProcessor, AutoModelForVision2Seq
from gptqmodel import GPTQModel, QuantizeConfig
from huggingface_hub import hf_hub_download
import struct
import numpy as np

from gptqmodel.utils.model_dequant import dequantize_model

from safetensors import safe_open
import os

import json
from transformers import AutoProcessor, AutoModelForVision2Seq


bin_path = hf_hub_download(
    repo_id='arashakb/llm-embedding-calib-data-libero',
    filename='spatial_10.bin',
    repo_type="dataset"  # 关键：这是 dataset，不是 model
)
def load_calib_bin(path):
    with open(path, "rb") as f:
        # -------- Header --------
        magic = f.read(16)
        version = struct.unpack("<I", f.read(4))[0]
        num_samples = struct.unpack("<I", f.read(4))[0]
        hidden_dim = struct.unpack("<I", f.read(4))[0]
        reserved = struct.unpack("<IIII", f.read(16))
        _ = struct.unpack("<I", f.read(4))  # padding

        print("MAGIC:", magic)
        print("Version:", version)
        print("Num samples:", num_samples)
        print("Hidden dim:", hidden_dim)

        # -------- Index block --------
        seq_lens = np.frombuffer(
            f.read(num_samples * 4),
            dtype=np.uint32
        )

        # -------- Offset block --------
        offsets = np.frombuffer(
            f.read(num_samples * 8),
            dtype=np.uint64
        )

        # -------- Data block --------
        data_start = f.tell()
        embeddings = []

        for i in range(num_samples):
            f.seek(data_start + int(offsets[i]))
            n_elem = seq_lens[i] * hidden_dim
            emb = np.frombuffer(
                f.read(n_elem * 4),
                dtype=np.float32
            ).reshape(seq_lens[i], hidden_dim)
            embeddings.append(emb)

    return {
        "magic": magic,
        "version": version,
        "hidden_dim": hidden_dim,
        "sequence_lengths": seq_lens,
        "embeddings": embeddings
    }
a = load_calib_bin(bin_path)


"""This following is for saving the language part of the VLA"""
# processor = AutoProcessor.from_pretrained("moojink/openvla-7b-oft-finetuned-libero-spatial", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     "moojink/openvla-7b-oft-finetuned-libero-spatial",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to("cuda:0")
#
# processor.tokenizer.save_pretrained("openvla7b-oft-libero_llm")
# vla.language_model.save_pretrained("openvla7b-oft-libero_llm")
#
# exit()


# pretrained_model_id = "openvla/openvla-7b"
# pretrained_model_id = "/home/Qitao/project/GPTQModel/openvla-7b-pytorch"
pretrained_model_id = "/home/Qitao/project/GPTQModel/openvla7b-oft-libero_llm"
# pretrained_model_id = "meta-llama/Llama-2-7b-chat-hf"

from transformers import AutoTokenizer, AutoModelForCausalLM


quantize_config = QuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
)

# load un-quantized model, the model will always be force loaded into cpu
model = GPTQModel.load(pretrained_model_id, quantize_config, trust_remote_code=True)


def get_emb(calibration_data, tokenizer, max_length=4096, calibration_data_sort='desc'):

    tokenized_calibration_data = []

    for text in calibration_data:
        tokenized = tokenizer(  # type: ignore[call-arg]
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        tokenized_calibration_data.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        })

    if calibration_data_sort == 'desc':
        calibration_dataset = sorted(
            tokenized_calibration_data,
            key=lambda item: item["input_ids"].shape[0],
            reverse=True,
        )
    else:
        calibration_dataset = sorted(
            tokenized_calibration_data,
            key=lambda item: item["input_ids"].shape[0],
        )

    output = []
    for each in calibration_dataset:
        input_ids = each['input_ids']
        embed_inputs = model.turtle_model.model.embed_tokens(input_ids)
        each['embed_inputs'] = embed_inputs.unsqueeze(0)

        output.append(each)

    return output



a = a['embeddings']
a = [torch.tensor(each).to(torch.bfloat16) for each in a]
calibration_dataset = [{'embed_inputs':each.unsqueeze(0)} for each in a]
model.quantize(calibration_dataset)
quant_path = "openvla7b-oft-gptqmodel-libero"
model.save(quant_path)
