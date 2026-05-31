from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Replace with your actual base model & adapter path
BASE_MODEL = "Qwen/Qwen3-4B"  
ADAPTER_PATH = "train_grpo_lora.py"

print("Loading model... (this may take 1-2 mins)")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.3, do_sample=True)

def chat():
    print("Chat started. Type 'quit' to exit.\n")
    history = []
    while True:
        user = input("You: ").strip()
        if user.lower() in ["quit", "exit"]: break
        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        out = pipe(prompt)[0]["generated_text"]
        # Extract only the assistant's response
        bot_reply = out.split("assistant\n")[-1].strip() if "assistant\n" in out else out.split(prompt)[-1].strip()
        print(f"Bot: {bot_reply}\n")
        history.append({"role": "assistant", "content": bot_reply})

chat()