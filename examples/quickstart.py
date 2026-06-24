from peft_doctor import create_safe_bnb_config, create_safe_lora_config, create_safe_training_args


def main() -> None:
    peft_config = create_safe_lora_config(model_name="meta-llama/Llama-3-8B", as_dict=True)
    bnb_config = create_safe_bnb_config(as_dict=True)
    training_args = create_safe_training_args()

    print("LoRA config:")
    print(peft_config)
    print()
    print("QLoRA config:")
    print(bnb_config)
    print()
    print("Training args:")
    print(training_args)


if __name__ == "__main__":
    main()
