"""
Preprocessing tab view
"""
import os
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from baseline.preprocessor.chunking_service import (
    FixedSizeChunkingStrategy,
    SlidingWindowChunkingStrategy,
    SentenceBasedChunkingStrategy,
    ParagraphBasedChunkingStrategy,
    SemanticChunkingStrategy,
    MarkdownHeaderChunkingStrategy
)
from baseline.retriever.retreiver import Retriever
from baseline.config.config import DOCUMENT_FOLDER_PATH

def render_processing_ui(selected_strategies):
    """Render the initial UI for the processing tab"""
    st.header("Document Preprocessing")
    
    # Show different message based on processing state
    if st.session_state.is_processing:
        st.info("⏳ Processing documents... Please wait.")
    else:
        st.write("This section allows you to preprocess documents using different chunking strategies.")
        st.write("Select a chunking strategy from the sidebar and click 'Process Documents' to begin.")
        
    # Display information about selected strategies
    st.subheader("Selected Strategies Information")
    
    if not selected_strategies:
        st.write("Please select at least one chunking strategy.")
    else:
        for strategy in selected_strategies:
            if strategy == "FixedSizeChunkingStrategy":
                st.write(f"**{strategy}**: Splits text into fixed-size chunks based on character count.")
            elif strategy == "SlidingWindowChunkingStrategy":
                st.write(f"**{strategy}**: Uses a sliding window approach with overlap between chunks.")
            elif strategy == "SentenceBasedChunkingStrategy":
                st.write(f"**{strategy}**: Splits text into chunks based on sentence boundaries.")
            elif strategy == "ParagraphBasedChunkingStrategy":
                st.write(f"**{strategy}**: Splits text into chunks based on paragraph boundaries.")
            elif strategy == "SemanticChunkingStrategy":
                st.write(f"**{strategy}**: Uses semantic understanding to create meaningful chunks.")
            elif strategy == "MarkdownHeaderChunkingStrategy":
                st.write(f"**{strategy}**: Splits Markdown text based on header structure.")


def process_documents(tab, selected_strategies, chunk_size, overlap):
    """Process documents using selected chunking strategies"""
    chunk_counts = {}
    processing_times = {}
    
    with tab:
        with st.spinner("Processing documents..."):
            # Check if strategies are selected
            if not selected_strategies:
                st.error("Please select at least one chunking strategy.")
                # Reset processing state
                st.session_state.is_processing = False
                return None
            
            # Clear previous processed strategies
            st.session_state.processed_strategies = []
            
            # Process each selected strategy
            for strategy_name in selected_strategies:
                st.text(f"Processing with {strategy_name}...")
                
                # Initialize the appropriate strategy with the correct parameters
                if strategy_name == "FixedSizeChunkingStrategy":
                    strategy = FixedSizeChunkingStrategy(chunk_size=chunk_size)
                elif strategy_name == "SlidingWindowChunkingStrategy":
                    strategy = SlidingWindowChunkingStrategy(chunk_size=chunk_size, overlap=overlap)
                elif strategy_name == "SentenceBasedChunkingStrategy":
                    strategy = SentenceBasedChunkingStrategy(chunk_size=chunk_size)
                elif strategy_name == "ParagraphBasedChunkingStrategy":
                    strategy = ParagraphBasedChunkingStrategy(chunk_size=chunk_size)
                elif strategy_name == "SemanticChunkingStrategy":
                    strategy = SemanticChunkingStrategy()
                elif strategy_name == "MarkdownHeaderChunkingStrategy":
                    strategy = MarkdownHeaderChunkingStrategy()
                
                # Create retriever and process documents
                retriever = Retriever(strategy)
                retriever.add_document(DOCUMENT_FOLDER_PATH, is_directory=True)
                
                # Track processing time
                start_time = time.time()
                chunks = retriever.preprocess()
                end_time = time.time()
                
                # Store chunk counts and processing time (convert to milliseconds)
                processing_time_sec = end_time - start_time
                processing_time_ms = processing_time_sec * 1000  # Convert to milliseconds
                chunk_counts[strategy_name] = len(chunks)
                processing_times[strategy_name] = processing_time_ms
                
                retriever.save()
                
                # Store the retriever in session state for reuse in interaction tab
                if 'strategy_retrievers' not in st.session_state:
                    st.session_state.strategy_retrievers = {}
                st.session_state.strategy_retrievers[strategy_name] = retriever
                
                # Store successfully processed strategies
                if strategy_name not in st.session_state.processed_strategies:
                    st.session_state.processed_strategies.append(strategy_name)
                
                st.text(f"✅ {strategy_name} completed: {len(chunks)} chunks created in {processing_time_ms:.0f} ms")
    
    # Store processing times in session state for future reference
    st.session_state.processing_times = processing_times
    return chunk_counts, processing_times


def load_insights_file():
    """Load insights file after processing"""
    try:
        # Get the directory where insight file should be stored
        insight_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "baseline", "insight")
        
        # Create insight directory if it doesn't exist
        if not os.path.exists(insight_dir):
            os.makedirs(insight_dir)
        
        # Check for both possible file name patterns
        insight_paths = [
            os.path.join(insight_dir, "chunking_strategy_insights.csv"),
            os.path.join(insight_dir, "chunking_strategy_insights.csv.csv")
        ]
        
        for path in insight_paths:
            if os.path.exists(path):
                # Actually load the data if it exists to update the app
                try:
                    insights_df = pd.read_csv(path)
                    st.session_state.insights_loaded = True
                    return insights_df
                except Exception:
                    pass
    except Exception as e:
        # Just log and continue even if reload fails
        print(f"Note: Failed to reload insights file: {str(e)}")
    
    return None


def display_processing_results(tab, results):
    """Display results after processing documents"""
    with tab:
        # Reset processing state and mark that we've processed at least once
        st.session_state.is_processing = False
        st.session_state.has_processed_once = True
        
        # Make sure processed_strategies is properly saved in session state
        if not st.session_state.processed_strategies:
            print("Warning: No processed strategies found in session state")
        else:
            print(f"Processed strategies in session state: {st.session_state.processed_strategies}")
            
        st.success(f"Document processing completed with strategies: {', '.join(st.session_state.processed_strategies)}! You can now use the Interaction tab to ask questions.")
        
        # Unpack the results
        chunk_counts, processing_times = results
        
        # Create and display the bar charts side by side
        if chunk_counts and processing_times:
            st.subheader("Processing Results")
            
            # Create two columns for side-by-side charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert data to DataFrame for chunk count plot
                df_chunks = pd.DataFrame({
                    'Strategy': list(chunk_counts.keys()),
                    'Chunk Count': list(chunk_counts.values())
                })
                
                # Create the chunk count bar chart
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                bars1 = ax1.bar(df_chunks['Strategy'], df_chunks['Chunk Count'], color='skyblue')
                
                # Add labels and title
                ax1.set_xlabel('Chunking Strategy')
                ax1.set_ylabel('Number of Chunks')
                ax1.set_title('Number of Chunks Created by Strategy')
                
                # Add values on top of the bars
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Display the chart in Streamlit
                st.pyplot(fig1)
            
            with col2:
                # Convert data to DataFrame for processing time plot
                df_times = pd.DataFrame({
                    'Strategy': list(processing_times.keys()),
                    'Processing Time (ms)': list(processing_times.values())
                })
                
                # Create the processing time bar chart
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bars2 = ax2.bar(df_times['Strategy'], df_times['Processing Time (ms)'], color='lightgreen')
                
                # Add labels and title
                ax2.set_xlabel('Chunking Strategy')
                ax2.set_ylabel('Time (milliseconds)')
                ax2.set_title('Processing Time by Strategy')
                
                # Add values on top of the bars
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}ms', ha='center', va='bottom')
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Display the chart in Streamlit
                st.pyplot(fig2)
