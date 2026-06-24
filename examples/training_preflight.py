from peft_doctor import diagnose_peft


def run_preflight(model, tokenizer, peft_config, training_args, train_dataset) -> None:
    report = diagnose_peft(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        training_args=training_args,
        train_dataset=train_dataset,
        sequence_length=2048,
    )

    print(report.to_markdown())
    if report.has_errors:
        raise SystemExit("Fix the PEFT Doctor errors before starting training.")
