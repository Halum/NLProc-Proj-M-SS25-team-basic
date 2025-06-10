def generate_answering_prompt(retrieved_chunks, query):
    context  = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful assistant. Use the following context to provide to the point answer of the question.

    Context:
    {context}

    Question:
    {query}

    Answer:"""
    
    return prompt

def generate_answer_evaluation_prompt(question, answer, expected_answer):
    prompt = f"""You are a critic in student's answer evaluation. Evaluate strictly and give a test score from 1 to 5.
    
    Question: 
    {question}
    
    Student Answer: 
    {answer}
    
    Expected Answer: 
    {expected_answer}
    
    Test Score:
    """
    
    return prompt