"""
Chat tab view for interactive conversation with RAG system
"""
import streamlit as st

from baseline.evaluation.answer_verifier import AnswerVerifier
from baseline.generator.generator import Generator
from baseline.evaluation.insight_generator import InsightGenerator

def render_chat_ui():
    """
    Render the chat interface UI
    
    Returns:
        tuple: (ask_button_clicked, message, selected_chat_strategies, has_insights)
    """
    # Function to handle the ask button callback
    def on_ask_button_click():
        st.session_state.ask_button_clicked = True
        
    # Initialize callback state if needed
    if "ask_button_clicked" not in st.session_state:
        st.session_state.ask_button_clicked = False
    
    st.header("Chat Interface")
    st.write("Have a conversation with the RAG system using different chunking strategies.")
    
    # Determine which strategies are available for interaction
    available_strategies, has_insights = get_available_strategies()
    
    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize pending insights if not present
    if "pending_insights" not in st.session_state:
        st.session_state.pending_insights = []
    
    # Initialize is_correct state per message if not present
    if "message_correctness" not in st.session_state:
        st.session_state.message_correctness = {}
    
    # Display chat history
    display_chat_history()
    
    # Create the message input interface with strategy selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Message input
        message = st.text_area("Your question:", key="chat_input", height=100)
        
    with col2:
        # Strategy selection - similar to interaction tab
        strategy_col = st.container()
        selected_chat_strategies = render_strategy_selector(strategy_col, available_strategies)
        
        # Add the Ask button
        st.write("")  # Add some space
        button_disabled = (
            st.session_state.is_processing or 
            not has_insights or 
            not selected_chat_strategies or
            not message  # Disable button if no message
        )
        ask_button = st.button(
            "Ask", 
            key="chat_ask_button", 
            disabled=button_disabled,
            on_click=on_ask_button_click  # Set the callback function
        )
    
    # Insight saving section
    if st.session_state.pending_insights:
        st.divider()
        st.subheader("Pending Insights")
        st.write(f"You have {len(st.session_state.pending_insights)} pending insights to save.")
        
        # Add button to save insights
        save_insights = st.button("Save All Insights", key="save_all_insights_button")
        
        if save_insights:
            with st.spinner("Saving insights..."):
                # Create insight generator
                insight_generator = InsightGenerator()
                
                # Add all pending insights
                for insight in st.session_state.pending_insights:
                    insight_generator.update_insight(
                        question=insight["question"],
                        retrieved_chunks=insight["retrieved_chunks"],
                        prompt=insight["prompt"],
                        generated_answer=insight["generated_answer"],
                        chunk_strategy=insight["chunk_strategy"],
                        number_of_chunks=insight["number_of_chunks"],
                        retrieved_chunk_rank=insight["retrieved_chunk_rank"],
                        is_correct_answer=st.session_state.message_correctness.get(insight["id"], True),
                        similarity_scores=insight["similarity_scores"],
                        similarity_mean=insight["similarity_mean"]
                    )
                
                # Save insights
                insight_generator.save_insight()
                
                # Clear pending insights
                st.session_state.pending_insights = []
                
                # Success message
                st.success("All insights saved successfully!")
                st.experimental_rerun()
    
    # Handle Ask button click from callback
    if st.session_state.ask_button_clicked:
        # Reset the flag
        st.session_state.ask_button_clicked = False
        
        # Store the current message
        current_message = message
        
        # Process the chat message with the selected strategies
        results_by_strategy = process_chat_message(current_message, selected_chat_strategies)
        
        # Add user message to history
        st.session_state.chat_history.append({
            "type": "user",
            "message": current_message
        })
        
        # Add system response to history
        st.session_state.chat_history.append({
            "type": "system",
            "responses": results_by_strategy
        })
        
        # Rerun to refresh UI
        st.experimental_rerun()
    
    return ask_button, message, selected_chat_strategies, has_insights


def get_available_strategies():
    """Determine which strategies are available for interaction
    (Same implementation as in interaction_tab.py)
    """
    # Only use strategies directly processed in the current session
    processed_from_session = st.session_state.processed_strategies
    
    # Check if we have retrievers stored in session state
    retriever_keys = list(st.session_state.strategy_retrievers.keys()) if 'strategy_retrievers' in st.session_state else []
    
    # Make sure we only show strategies that have retrievers available
    # In case they somehow got out of sync
    valid_strategies = [s for s in processed_from_session if s in retriever_keys]
    
    # Check if we have any processed strategies with available retrievers
    if valid_strategies:
        available_strategies = sorted(valid_strategies)
        has_insights = True
        
        # If we're actively processing, update the UI
        if st.session_state.is_processing:
            st.info("Processing documents... Chat tab will be updated when finished.")
            
        # If we've processed before and switched tabs, restore the success message
        elif st.session_state.has_processed_once:
            st.success(f"Document processing completed with {len(valid_strategies)} strategies. You can now ask questions.")
    else:
        # No strategies available yet or retrievers missing
        available_strategies = []
        has_insights = False
        
        if processed_from_session and not valid_strategies:
            # Strategies were processed but retrievers are missing - error state
            st.error("Processed strategies exist but their retrievers are missing. Please reprocess the documents.")
        elif st.session_state.selected_strategies and not st.session_state.is_processing:
            # User has selected strategies but hasn't processed them
            st.warning("You've selected strategies in the sidebar but haven't processed them yet. Please click 'Process Documents' button in the sidebar first.")
        else:
            # Generic message
            st.warning("Please process documents first using the sidebar options.")
    
    return available_strategies, has_insights


def render_strategy_selector(container, available_strategies):
    """Render the strategy selection as checkboxes 
    (Simplified version of the one in interaction_tab.py)
    """
    with container:
        # Make sure we have a valid selection if strategies are available
        if available_strategies:
            # If we don't have any selected strategies yet or our selections are no longer valid
            if not st.session_state.selected_chat_strategies:
                # Default to first available strategy
                st.session_state.selected_chat_strategies = [available_strategies[0]]
            else:
                # Only keep valid selections
                valid_selections = [s for s in st.session_state.selected_chat_strategies 
                                   if s in available_strategies]
                
                # If we lost all valid selections, default to first available
                if not valid_selections and available_strategies:
                    valid_selections = [available_strategies[0]]
                    
                st.session_state.selected_chat_strategies = valid_selections
        
        # Create checkboxes for strategies (only show used strategies)
        if available_strategies:
            # Show a note about processed strategies
            st.write("**Select strategies:**")
            
            # Create a checkbox for each strategy
            selected_chat_strategies = []
            for strategy in available_strategies:
                # Initialize checkbox state if not already in session state
                if f"chat_strategy_{strategy}" not in st.session_state:
                    st.session_state[f"chat_strategy_{strategy}"] = strategy in st.session_state.selected_chat_strategies
                
                # Create the checkbox
                is_selected = st.checkbox(
                    strategy, 
                    key=f"chat_strategy_{strategy}",
                    value=st.session_state[f"chat_strategy_{strategy}"]
                )
                
                # If selected, add to the list
                if is_selected:
                    selected_chat_strategies.append(strategy)
            
            # Update the selected_chat_strategies in session state
            st.session_state.selected_chat_strategies = selected_chat_strategies
        else:
            # Display message to guide user
            st.info("Process documents with selected strategies in the sidebar first")
            selected_chat_strategies = []
            
        return selected_chat_strategies


def process_chat_message(message, selected_chat_strategies):
    """Process the chat message with selected strategies
    
    Args:
        message: The user message to process
        selected_chat_strategies: List of strategies to use
        
    Returns:
        Dictionary mapping strategies to their processing results
    """
    results_by_strategy = {}
    
    # Process the message for each selected strategy
    for selected_strategy in selected_chat_strategies:
        with st.spinner(f"Processing with {selected_strategy}..."):
            try:
                # First check if the strategy has actually been processed
                if selected_strategy not in st.session_state.processed_strategies:
                    st.error(f"{selected_strategy} has not been processed. Please go to the Preprocessing tab and process this strategy first.")
                    continue
                
                # Use the pre-processed retriever from session state if available
                if 'strategy_retrievers' in st.session_state and selected_strategy in st.session_state.strategy_retrievers:
                    # Use existing retriever that was stored during document processing
                    retriever = st.session_state.strategy_retrievers[selected_strategy]
                    
                    # Check if the retriever is a valid object
                    if retriever is not None:
                        st.success(f"Using pre-processed retriever for {selected_strategy}")
                    else:
                        st.error(f"Retriever for {selected_strategy} appears to be invalid. Please reprocess the documents.")
                        continue
                else:
                    # If for some reason the retriever isn't available, show an error
                    st.error(f"Pre-processed data for {selected_strategy} is missing. Please reprocess the documents.")
                    continue
                
                # Load the processed chunks (include distances for similarity scores)
                retrieved_chunks, distances = retriever.query(message)
                
                # Generate answer
                answering_prompt = Generator.build_answering_prompt(message, retrieved_chunks)
                generated_answer = Generator.generate_answer(answering_prompt)
                
                # Create a unique ID for this message-strategy combination
                message_id = f"{len(st.session_state.chat_history)}_{selected_strategy}"
                
                # Initialize the message correctness if not already set
                if message_id not in st.session_state.message_correctness:
                    st.session_state.message_correctness[message_id] = True
                
                # Create an insight record (to be saved later)
                insight = {
                    "id": message_id,
                    "question": message,
                    "retrieved_chunks": retrieved_chunks,
                    "prompt": answering_prompt,
                    "generated_answer": generated_answer,
                    "chunk_strategy": selected_strategy,
                    "number_of_chunks": st.session_state.chunk_counts.get(selected_strategy, 0) if hasattr(st.session_state, 'chunk_counts') else len(retriever.chunks),
                    "retrieved_chunk_rank": -1,  # No expected chunk in chat
                    "similarity_scores": distances[0],
                    "similarity_mean": distances[0].mean()
                }
                
                # Add to pending insights
                st.session_state.pending_insights.append(insight)
                
                # Store results for this strategy
                results_by_strategy[selected_strategy] = {
                    "message_id": message_id,
                    "generated_answer": generated_answer,
                    "retrieved_chunks": retrieved_chunks,
                    "distances": distances
                }
                
            except Exception as e:
                st.error(f"Error processing message with {selected_strategy}: {str(e)}")
    
    return results_by_strategy


def display_chat_history():
    """Display the chat history in the UI"""
    chat_container = st.container()
    
    with chat_container:
        for i, chat_entry in enumerate(st.session_state.chat_history):
            message_type = chat_entry.get("type", "user")
            
            if message_type == "user":
                # User message
                st.markdown(f"🧑‍💻 **You:** {chat_entry['message']}")
            
            elif message_type == "system":
                # System message with potentially multiple answers
                st.markdown("🤖 **AI:**")
                
                # Create tabs for each strategy if there are multiple
                strategies = list(chat_entry['responses'].keys())
                
                if len(strategies) > 1:
                    # Multiple strategies - use tabs
                    strategy_tabs = st.tabs(strategies)
                    
                    # Display results in tabs
                    for i, strategy_name in enumerate(strategies):
                        result = chat_entry['responses'][strategy_name]
                        message_id = result.get("message_id", f"msg_{i}")
                        
                        with strategy_tabs[i]:
                            # Display the answer
                            st.markdown(result["generated_answer"])
                            
                            # Add feedback checkbox
                            is_correct = st.checkbox(
                                "This answer is correct", 
                                value=st.session_state.message_correctness.get(message_id, True),
                                key=f"correct_{message_id}"
                            )
                            
                            # Update correctness in session state
                            st.session_state.message_correctness[message_id] = is_correct
                            
                            # Show retrieved chunks in expander
                            with st.expander("Retrieved Chunks", expanded=False):
                                for j, chunk in enumerate(result["retrieved_chunks"]):
                                    st.markdown(f"**Chunk {j+1}:**")
                                    # Use a container with CSS styling to wrap text
                                    st.markdown(
                                        f"<div style='white-space: pre-wrap; overflow-wrap: break-word; max-width: 100%;'>{chunk}</div>", 
                                        unsafe_allow_html=True
                                    )
                                    st.markdown("---")
                else:
                    # Just one strategy
                    strategy_name = strategies[0]
                    result = chat_entry['responses'][strategy_name]
                    message_id = result.get("message_id", f"msg_{i}")
                    
                    # Display the answer
                    st.markdown(result["generated_answer"])
                    
                    # Add feedback checkbox
                    is_correct = st.checkbox(
                        "This answer is correct", 
                        value=st.session_state.message_correctness.get(message_id, True),
                        key=f"correct_{message_id}"
                    )
                    
                    # Update correctness in session state
                    st.session_state.message_correctness[message_id] = is_correct
                    
                    # Show retrieved chunks in expander
                    with st.expander("Retrieved Chunks", expanded=False):
                        for j, chunk in enumerate(result["retrieved_chunks"]):
                            st.markdown(f"**Chunk {j+1}:**")
                            # Use a container with CSS styling to wrap text
                            st.markdown(
                                f"<div style='white-space: pre-wrap; overflow-wrap: break-word; max-width: 100%;'>{chunk}</div>", 
                                unsafe_allow_html=True
                            )
                            st.markdown("---")
            
            # Add a divider between conversations
            st.divider()
