# RAG Chunking Strategies Explorer

This is a Streamlit application for exploring different chunking strategies for Retrieval-Augmented Generation (RAG).

## Changelog

### May 2023
- **Added**: Process button is now disabled during document processing to prevent duplicate processing
- **Added**: Processed strategies are automatically available in the Interaction tab
- **Fixed**: Proper handling of strategy selection in the Interaction tab based on processed strategies
- **Enhanced**: Better error handling for insight file loading

### April 2023
- **Fixed**: Resolved issue with multiselect widgets requiring double clicks to select or remove items by directly connecting widgets to session state
- **Enhanced**: Improved session state management for all interactive components with a simplified approach
- **Added**: Persistence of user selections between interactions without disrupting the UI experience

## Features

- View and select multiple chunking strategies from the sidebar
- Dynamically adjust parameters like chunk size and overlap based on selected strategies
- Process documents using multiple chunking strategies with a single button click
- Visualize the number of chunks created by each strategy through bar charts
- View detailed information about each chunking strategy
- Query functionality with predefined sample questions
- Compare results across multiple strategies simultaneously
- Visualization of chunking strategy performance metrics

## Setup & Running

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   cd /Users/sajjadhossain/Documents/01\ -\ Projects/05\ MSc/nlp-project/src
   streamlit run streamlit/app.py
   ```

3. The app should open in your default web browser. If not, navigate to the URL displayed in the terminal (usually http://localhost:8501).

## Usage

### Preprocessing Tab

1. Select one or more chunking strategies from the multiselect dropdown in the sidebar:
   - Choose any combination of available strategies to compare their performance
   - Parameter sliders will appear or hide based on your selections

2. Adjust parameters as needed:
   - Chunk size slider appears when strategies that use fixed-size chunks are selected
   - Overlap slider appears when the sliding window strategy is selected

3. Click the "Process Documents" button to start processing
   - This will load documents from the configured folder 
   - Apply all selected chunking strategies
   - Display the results including number of chunks created by each strategy

4. View the bar chart showing the number of chunks created by each strategy

### Interaction Tab

1. Select a query from the dropdown menu of predefined sample questions
2. Choose one or more chunking strategies that have been processed
3. Click the "Ask" button to execute the query
4. View the results in tabbed format (one tab per selected strategy):
   - Generated answer vs. expected answer
   - Context from the source document
   - Whether the context was found in the retrieved chunks
   - Expandable view of all retrieved chunks
5. After executing a query, explore performance metrics:
   - Number of chunks by strategy (smaller graphs displayed side by side)
   - Correct answer rate by strategy
   - Context found rate by strategy
   - Raw data in a compact format

## Available Chunking Strategies

- ALL: Process documents using all available strategies
- FixedSizeChunkingStrategy: Splits text into fixed-size chunks based on character count
- SlidingWindowChunkingStrategy: Uses a sliding window approach with overlap between chunks
- SentenceBasedChunkingStrategy: Splits text into chunks based on sentence boundaries
- ParagraphBasedChunkingStrategy: Splits text into chunks based on paragraph boundaries
- SemanticChunkingStrategy: Uses semantic understanding to create meaningful chunks
- MarkdownHeaderChunkingStrategy: Splits Markdown text based on header structure
