labeled_data = [
{
    "query": "Who are the main countries involved in the commercial hostilities discussed in the text?",
    "answer": "The United States and China",
    "context": "In recent months, the open commercial hostilities between the United States of America and the People's Republic of China"
},
{
    "query": "According to the text, who is the European Union's main exporting partner, and who is its largest import source?",
    "answer": "The United States is the main exporting partner, and China is the largest import source.",
    "context": "The USA was the main place of export from the EU, while China was the first place of import."
},
{
    "query": "What could the European Union potentially benefit from if the trade war continues between the US and China?",
    "answer": "The European Union could potentially benefit from the US-China trade war by gaining market share, attracting trade and investment diverted from both countries, and increasing exports due to its neutral position—potentially earning up to $70 billion according to UN estimates.",
    "context": "If the trade war continues and the positions of the US and China harden, the EU, as the main partner of both, could receive benefits thanks to a redistribution of the flow of trade."
},
{
    "query": "How does the text describe the role of the European Union in relation to the US and China amid the trade conflict?",
    "answer": "The European Union is a key partner for both the US and China, with the potential to emerge stronger if it maintains neutrality and avoids tariff impositions.",
    "context": "The EU, as the main partner of both, could receive benefits thanks to a redistribution of the flow of trade. So, to avoid the loss due to tariffs, both China and the US could sell products with heavy taxes in the other country to the European market..."
},
{
    "query": "What strategy does the European Union need to adopt to benefit from the US-China trade war, according to the text?",
    "answer": "The EU must remain neutral and not lean towards either the US or China, ensuring it doesn't impose tariffs on its own products.",
    "context": "Alicia García-Herrero... affirms that the benefit for Europe will only be possible if it does not lean towards any of the contenders and remains neutral on the economic level."
}]


class AnswerVerifier:
    """
    Class for verifying and evaluating the correctness of generated answers
    by comparing them with labeled data and context.
    """
    
    @staticmethod
    def get_sample_labeled_data():
        """
        Retrieve sample labeled data for testing and evaluation purposes.
        
        Returns:
            dict: A dictionary containing query, expected answer and context.
        """
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
        
        return -1, None