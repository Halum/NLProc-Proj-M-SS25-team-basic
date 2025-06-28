"""
Central module for defining available chunking strategies and their abbreviations.
This ensures consistency across all views in the application.
"""

# List of available chunking strategies (should match what's used in app.py)
AVAILABLE_STRATEGIES = [
    "FixedSizeChunkingStrategy",
    "SlidingWindowChunkingStrategy",
    "SentenceBasedChunkingStrategy", 
    "ParagraphBasedChunkingStrategy",
    "SemanticChunkingStrategy",
    "MarkdownHeaderChunkingStrategy"
]

# Dictionary mapping full strategy names to abbreviations
STRATEGY_ABBREVIATIONS = {
    'FixedSizeChunkingStrategy': 'FSCS',
    'SlidingWindowChunkingStrategy': 'SWCS',
    'SentenceBasedChunkingStrategy': 'SBCS',
    'ParagraphBasedChunkingStrategy': 'PBCS',
    'SemanticChunkingStrategy': 'SCS',
    'MarkdownHeaderChunkingStrategy': 'MHCS'
}

def get_short_strategy_name(strategy_name):
    """
    Convert long strategy names to short abbreviations
    
    Args:
        strategy_name: Original strategy name
        
    Returns:
        Short name/abbreviation for the strategy
    """
    # Check if the strategy name exists in our mapping
    if strategy_name in STRATEGY_ABBREVIATIONS:
        return STRATEGY_ABBREVIATIONS[strategy_name]
    
    # For strategies not in the dictionary, create an abbreviation using first letters
    if strategy_name:
        # Try to identify camel case or other naming patterns
        words = []
        current_word = ""
        for char in strategy_name:
            if char.isupper() and current_word:
                words.append(current_word)
                current_word = char
            else:
                current_word += char
        if current_word:
            words.append(current_word)
        
        # If we couldn't split by camel case, try splitting by spaces or underscores
        if len(words) <= 1:
            words = strategy_name.replace('_', ' ').split()
        
        # Create abbreviation from first letters
        abbreviation = ''.join(word[0].upper() for word in words if word)
        return abbreviation
    
    # Return the original if we can't create an abbreviation
    return strategy_name
