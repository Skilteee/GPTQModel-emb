import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from gptqmodel import GPTQModel, QuantizeConfig

# pretrained_model_id = "openvla/openvla-7b"
pretrained_model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(100))["text"]


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


with torch.no_grad():
    calibration_dataset = get_emb(calibration_dataset, tokenizer)

model.quantize(calibration_dataset)
