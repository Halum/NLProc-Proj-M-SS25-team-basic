labeled_data = {
    "query": "Who are the main countries involved in the commercial hostilities discussed in the text?",
    "answer": "The United States and China",
    "context": "In recent months, the open commercial hostilities between the United States of America and the People's Republic of China"
}


class AnswerVerifier:
    @staticmethod
    def get_sample_labeled_data():
        return labeled_data
    
    @staticmethod
    def find_chunk_containing_context(retrieved_chunks, context):
        """
        Find the first chunk that contains the specified context.
        
        Args:
            retrieved_chunks: List of text chunks to search through
            context: The context text to look for
            
        Returns:
            tuple: (index, chunk) of the first matching chunk, or (None, None) if not found
        """
        for i, chunk in enumerate(retrieved_chunks):
            if context in chunk:
                return i, chunk
        
        return None, None