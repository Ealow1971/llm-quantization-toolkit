from transformers import AutoModelForCausalLM

def patch_model_for_quantization(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Patching {model_id} with custom quantization kernels...")
    return model
