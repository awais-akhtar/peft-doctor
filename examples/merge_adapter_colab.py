from peft_doctor import merge_lora_adapter


def main() -> None:
    result = merge_lora_adapter(
        base_model="meta-llama/Llama-2-7b-hf",
        adapter="your-user/your-lora-adapter",
        output_dir="merged-model",
        torch_dtype="fp16",
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
