from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
)

class LLM:
    @staticmethod
    def generate_embedding(chunks):
        embedding_model = LLM.embedding_model()
        embeddings = embedding_model.embed_documents(chunks)
        
        return embeddings
    
    @staticmethod
    def embedding_dimensions():
        embedding_model = LLM.embedding_model()
        
        return embedding_model.client.get_sentence_embedding_dimension()
    
    @staticmethod
    def embedding_model():
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        return embedding_model
    
    @staticmethod
    def generate_answering_prompt(query, relevant_contexts):
        context  = "\n\n".join(relevant_contexts)
        
        prompt = f"""You are a helpful assistant. Use the following context to provide to the point answer of the question.

        Context:
        {context}

        Question:
        {query}

        Answer:"""
        
        return prompt
    
    @staticmethod
    def invoke_llm(prompt):
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, low_cpu_mem_usage=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
            
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer