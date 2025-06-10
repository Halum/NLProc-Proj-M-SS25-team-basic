from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config import (
    LLM_MODEL,
)

def invoke_llm(prompt):
    # device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, low_cpu_mem_usage=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
        
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer
