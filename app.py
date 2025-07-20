import streamlit as st
import google.generativeai as genai
import numpy as np
import pickle
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configure page
st.set_page_config(
    page_title="Advanced AI Chatbot Demo",
    page_icon="🦄",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorized_memory" not in st.session_state:
    st.session_state.vectorized_memory = []
if "memory_vectors" not in st.session_state:
    st.session_state.memory_vectors = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
if "model" not in st.session_state:
    st.session_state.model = None

# Sidebar for configuration
with st.sidebar:
    st.title("🦄 AI Techniques Demo")
    
    # API Key input
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.model = genai.GenerativeModel('gemma-3n-e4b-it')
            st.success("✅ API Key configured!")
        except Exception as e:
            st.error(f"❌ API Error: {str(e)}")
    
    st.divider()
    
    # Technique indicators
    st.subheader("Active Techniques")
    st.info("🧠 **Fine Tuned Model**: Using gemma-3n-e4b-it")
    st.info("📚 **RAG**: Retrieval from conversation history")
    st.info("🔍 **Vectorized Memory**: TF-IDF similarity search")
    st.info("📝 **Text Compression**: Dynamic prompt optimization")
    st.info("💾 **Large Codebase Fitting**: Context management")
    
    # Memory stats
    if st.session_state.vectorized_memory:
        st.metric("Memory Entries", len(st.session_state.vectorized_memory))
        
    # Clear memory
    if st.button("🗑️ Clear Memory"):
        st.session_state.vectorized_memory = []
        st.session_state.memory_vectors = None
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🦄 Advanced AI Chatbot")
st.markdown("*Demonstrating: Fine-tuning, RAG, Vectorized Memory, Text Compression & Large Codebase Management*")

# Technique explanation
with st.expander("🔬 How it works", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🧠 Fine-tuned Model**: gemma-3n-e4b-it optimized for instruction following
        
        **📚 RAG (Retrieval Augmented Generation)**: Searches conversation history for relevant context
        
        **🔍 Vectorized Memory**: Uses TF-IDF vectorization for semantic similarity matching
        """)
    with col2:
        st.markdown("""
        **📝 Text Compression**: Compresses long contexts while preserving key information
        
        **💾 Large Codebase Management**: Handles extensive conversation history efficiently
        """)

def compress_text(text, max_length=200):
    """Text compression technique - keeps key information"""
    if len(text) <= max_length:
        return text
    
    # Keep first and last parts, compress middle
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) <= 2:
        return text[:max_length] + "..."
    
    first_part = sentences[0]
    last_part = sentences[-1]
    middle_compressed = f"...[{len(sentences)-2} sentences compressed]..."
    
    compressed = f"{first_part}.{middle_compressed}.{last_part}"
    return compressed if len(compressed) <= max_length else text[:max_length] + "..."

def add_to_vectorized_memory(query, response):
    """Add conversation to vectorized memory"""
    memory_entry = {
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "combined_text": f"Q: {query} A: {response}"
    }
    
    st.session_state.vectorized_memory.append(memory_entry)
    
    # Update vectors
    if len(st.session_state.vectorized_memory) >= 1:
        texts = [entry["combined_text"] for entry in st.session_state.vectorized_memory]
        try:
            st.session_state.memory_vectors = st.session_state.vectorizer.fit_transform(texts)
        except:
            pass

def retrieve_relevant_context(query, top_k=3):
    """RAG implementation - retrieve relevant context from memory"""
    if not st.session_state.vectorized_memory or st.session_state.memory_vectors is None:
        return []
    
    try:
        query_vector = st.session_state.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, st.session_state.memory_vectors)[0]
        
        # Get top_k most similar entries
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Similarity threshold
                entry = st.session_state.vectorized_memory[idx]
                relevant_contexts.append({
                    "text": entry["combined_text"],
                    "similarity": similarities[idx],
                    "timestamp": entry["timestamp"]
                })
        
        return relevant_contexts
    except:
        return []

def manage_large_context(messages, max_context_length=4000):
    """Large codebase fitting - manage conversation context"""
    # Calculate total length
    total_length = sum(len(msg["content"]) for msg in messages)
    
    if total_length <= max_context_length:
        return messages
    
    # Keep recent messages and compress older ones
    recent_messages = messages[-5:]  # Keep last 5 messages
    older_messages = messages[:-5]
    
    if older_messages:
        # Compress older messages
        compressed_summary = "Previous conversation summary: "
        for msg in older_messages[-3:]:  # Summarize last 3 of older messages
            compressed_summary += compress_text(msg["content"], 100) + " | "
        
        summary_message = {
            "role": "assistant",
            "content": compressed_summary,
            "compressed": True
        }
        
        return [summary_message] + recent_messages
    
    return recent_messages

def generate_response(user_input):
    """Generate response using all techniques"""
    if not st.session_state.model:
        return "Please configure your Gemini API key first."
    
    try:
        # 1. RAG - Retrieve relevant context
        relevant_contexts = retrieve_relevant_context(user_input)
        
        # 2. Build context with compression
        context = ""
        if relevant_contexts:
            context = "\nRelevant previous discussions:\n"
            for ctx in relevant_contexts:
                compressed_ctx = compress_text(ctx["text"], 150)
                context += f"- {compressed_ctx} (similarity: {ctx['similarity']:.2f})\n"
        
        # 3. Manage large conversation context
        managed_messages = manage_large_context(st.session_state.messages)
        
        # 4. Build optimized prompt
        system_prompt = f"""You are an advanced AI assistant demonstrating multiple AI techniques:
        - Fine-tuned response generation using gemma-3n-e4b-it
        - RAG-based context retrieval  
        - Vectorized memory search
        - Text compression for efficiency
        - Large context management
        
        Be helpful and mention which techniques you're using when relevant.
        {context}
        
        Current conversation context has been optimized for efficiency."""
        
        # 5. Generate response with fine-tuned model
        conversation_context = ""
        for msg in managed_messages[-3:]:  # Use last 3 messages for context
            if not msg.get("compressed"):
                conversation_context += f"{msg['role']}: {msg['content']}\n"
        
        full_prompt = f"{system_prompt}\n\nConversation:\n{conversation_context}\nUser: {user_input}\nAssistant:"
        
        response = st.session_state.model.generate_content(full_prompt)
        
        # Add transparency about techniques used
        technique_info = "\n\n---\n**🔬 Techniques Applied:**\n"
        technique_info += f"• Fine-tuned Model: gemma-3n-e4b-it\n"
        technique_info += f"• RAG: Retrieved {len(relevant_contexts)} relevant contexts\n"
        technique_info += f"• Memory: {len(st.session_state.vectorized_memory)} entries in vectorized memory\n"
        technique_info += f"• Compression: Context managed to fit {len(managed_messages)} messages\n"
        
        full_response = response.text + technique_info
        
        # 6. Add to vectorized memory for future RAG
        add_to_vectorized_memory(user_input, response.text)
        
        return full_response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Example prompts section
st.subheader("🎯 Demo Prompts")
with st.expander("Try these prompts to see all techniques in action", expanded=False):
    example_prompts = [
        "Explain machine learning algorithms and their applications",
        "What did we discuss about machine learning earlier? Build upon that conversation",
        "Create a detailed technical architecture for a distributed system with microservices",
        "Compare our previous discussions about AI techniques and suggest improvements",
        "Write a comprehensive guide on database optimization strategies"
    ]
    
    st.markdown("**Suggested conversation flow:**")
    for i, prompt in enumerate(example_prompts, 1):
        if st.button(f"{i}. {prompt}", key=f"example_{i}"):
            st.session_state.example_input = prompt
    
    st.markdown("*Each prompt will demonstrate different aspects of the AI techniques*")

# Chat interface
for message in st.session_state.messages:
    if not message.get("compressed"):  # Don't display compressed messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle example input
if "example_input" in st.session_state:
    user_input = st.session_state.example_input
    del st.session_state.example_input
else:
    user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (applying AI techniques)"):
            response = generate_response(user_input)
            st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.caption("🚀 This demo showcases advanced AI techniques including fine-tuning, RAG, vectorized memory, text compression, and large codebase management in a single chatbot interface.")
