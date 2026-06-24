from peft_doctor import infer_model_family, recommend_target_modules


class Leaf:
    def children(self):
        return []


class Block:
    def __init__(self):
        self.q_proj = Leaf()

    def children(self):
        return [self.q_proj]


class DummyModel:
    def named_modules(self):
        names = [
            ("", self),
            ("layers.0.self_attn.q_proj", Leaf()),
            ("layers.0.self_attn.k_proj", Leaf()),
            ("layers.0.self_attn.v_proj", Leaf()),
            ("layers.0.self_attn.o_proj", Leaf()),
            ("layers.0.mlp.gate_proj", Leaf()),
            ("layers.0.mlp.up_proj", Leaf()),
            ("layers.0.mlp.down_proj", Leaf()),
        ]
        return iter(names)


def test_infer_model_family_from_name():
    assert infer_model_family(model_name="meta-llama/Llama-3-8B") == "llama"
    assert infer_model_family(model_name="Qwen/Qwen2.5-7B") == "qwen2"
    assert infer_model_family(model_name="gpt2") == "gpt2"


def test_recommend_target_modules_from_family():
    assert recommend_target_modules(model_family="gpt2") == ["c_attn", "c_proj", "c_fc"]


def test_recommend_target_modules_from_model():
    assert recommend_target_modules(model=DummyModel(), model_family="llama") == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
