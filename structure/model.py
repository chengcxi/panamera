import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from PIL import Image

model_id = "openbmb/MiniCPM-V-4.6"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def minicpm_chat(user_text: str, image: Image = None) -> str:
    # Build the conversation in MiniCPM‑V’s format.
    # For text‑only, use the normal chat template.
    # For image+text, you must insert the image placeholder.
    messages = []
    if image is None:
        messages.append({"role": "user", "content": user_text})
    else:
        # MiniCPM‑V‑4.6 expects: "<image>\n{text}"
        messages.append({"role": "user", "content": f"<image>\n{user_text}"})

    # Tokenize and generate
    if image is not None:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        # The processor step: MiniCPM‑V expects the image separately
        # and the tokenizer inserts a placeholder.
        # We use model’s chat method which internally handles image:
        response, _ = model.chat(
            image=image,
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
        )
    else:
        response, _ = model.chat(
            image=None,
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
        )
    return response