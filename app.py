import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
import os
import re
from datetime import datetime
import zipfile
import tempfile
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Coding Assistant - Final Project",
    page_icon="🤖",
    layout="wide"
)

class CodebaseAnalyzer:
    """Handles large codebase processing and context window optimization"""
    
    def __init__(self):
        self.file_cache = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def extract_files_from_zip(self, zip_file):
        """Extract and process files from uploaded zip"""
        files = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
            for root, dirs, filenames in os.walk(temp_dir):
                for filename in filenames:
                    if filename.endswith(('.py', '.js', '.java', '.cpp', '.c', '.h', '.md', '.txt')):
                        filepath = os.path.join(root, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                relative_path = os.path.relpath(filepath, temp_dir)
                                files[relative_path] = content
                        except:
                            continue
        return files
    
    def chunk_code(self, code, max_chunk_size=2000):
        """Intelligent code chunking for context window optimization"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
                
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    
    def compress_prompt(self, text, compression_ratio=0.5):
        """Text-to-prompt compression using extractive summarization"""
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) <= 3:
            return text
            
        # Simple extractive summarization
        sentence_vectors = self.vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Score sentences based on similarity to others
        scores = similarity_matrix.sum(axis=1)
        top_indices = np.argsort(scores)[::-1][:int(len(sentences) * compression_ratio)]
        
        compressed_sentences = [sentences[i] for i in sorted(top_indices)]
        return '. '.join(compressed_sentences)

class VectorizedMemory:
    """RAG system for code snippets and documentation"""
    
    def __init__(self):
        self.memory_store = {}
        self.vectors = {}
        self.vectorizer = TfidfVectorizer(max_features=500)
        
    def add_memory(self, key, content, metadata=None):
        """Add content to vectorized memory"""
        self.memory_store[key] = {
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        self._update_vectors()
    
    def _update_vectors(self):
        """Update vector representations of stored content"""
        if not self.memory_store:
            return
            
        contents = [item['content'] for item in self.memory_store.values()]
        self.vectors = self.vectorizer.fit_transform(contents)
    
    def search_similar(self, query, top_k=3):
        """Search for similar content using cosine similarity"""
        if not self.memory_store:
            return []
            
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        keys = list(self.memory_store.keys())
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                key = keys[idx]
                results.append({
                    'key': key,
                    'content': self.memory_store[key]['content'],
                    'score': similarities[idx],
                    'metadata': self.memory_store[key]['metadata']
                })
        
        return results

class PromptOptimizer:
    """Fine-tuning strategies and prompt optimization"""
    
    @staticmethod
    def create_coding_prompt(task, context="", similar_examples=None, style="professional"):
        """Create optimized prompts for coding tasks"""
        
        style_guides = {
            "professional": "Write clean, well-documented, production-ready code.",
            "beginner": "Write simple, well-commented code with explanations.",
            "expert": "Write efficient, optimized code using advanced techniques.",
            "educational": "Write code with detailed explanations and learning notes."
        }
        
        prompt = f"""You are an expert software engineer. {style_guides.get(style, style_guides['professional'])}

TASK: {task}

"""
        
        if context:
            prompt += f"CONTEXT:\n{context}\n\n"
        
        if similar_examples:
            prompt += "SIMILAR EXAMPLES:\n"
            for i, example in enumerate(similar_examples[:2], 1):
                prompt += f"Example {i}:\n{example['content'][:500]}...\n\n"
        
        prompt += """REQUIREMENTS:
- Include error handling
- Add appropriate comments
- Follow best practices
- Make code modular and reusable

Please provide the solution:"""
        
        return prompt
    
    @staticmethod
    def create_analysis_prompt(code, analysis_type="general"):
        """Create prompts for code analysis tasks"""
        
        analysis_types = {
            "general": "Analyze this code for overall quality, structure, and improvements.",
            "performance": "Focus on performance optimization and efficiency improvements.",
            "security": "Identify potential security vulnerabilities and suggest fixes.",
            "maintainability": "Evaluate code maintainability and suggest refactoring opportunities."
        }
        
        return f"""As a senior code reviewer, {analysis_types.get(analysis_type, analysis_types['general'])}

CODE TO ANALYZE:
```
{code}
```

Please provide:
1. Summary of findings
2. Specific issues identified
3. Recommended improvements
4. Code examples for fixes (if applicable)
"""

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = VectorizedMemory()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = CodebaseAnalyzer()
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar for API key and settings
st.sidebar.title("🔧 Configuration")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key:",
    type="password",
    help="Get your API key from https://makersuite.google.com/"
)

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemma-3-27b-it')
    st.sidebar.success("API Key configured!")
else:
    st.sidebar.warning("Please enter your Gemini API Key")

st.sidebar.markdown("---")

# Model settings
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 100, 8000, 2000)
coding_style = st.sidebar.selectbox(
    "Coding Style",
    ["professional", "beginner", "expert", "educational"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Memory Stats")
st.sidebar.metric("Stored Memories", len(st.session_state.memory.memory_store))

# Main interface
st.title("🤖 AI Coding Assistant - Final Project")
st.markdown("*Showcasing LLM deployment, RAG, vectorized memory, text-to-prompt compression, and fine-tuning strategies*")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat Assistant", 
    "📁 Codebase Analysis", 
    "🧠 Memory Management", 
    "🔍 Code Review",
    "📚 Knowledge Base"
])

with tab1:
    st.header("AI Coding Assistant Chat")
    
    if not api_key:
        st.warning("Please configure your Gemini API key in the sidebar to start chatting.")
    else:
        # Chat interface
        user_input = st.text_area("Ask me anything about coding:", height=100)
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            send_button = st.button("Send", type="primary")
        with col2:
            use_rag = st.checkbox("Use RAG", value=True)
        
        if send_button and user_input:
            try:
                # Search for similar content if RAG is enabled
                context = ""
                similar_examples = []
                
                if use_rag:
                    similar_examples = st.session_state.memory.search_similar(user_input)
                    if similar_examples:
                        context = f"Related information from memory:\n"
                        for example in similar_examples[:2]:
                            context += f"- {example['content'][:200]}...\n"
                
                # Create optimized prompt
                optimizer = PromptOptimizer()
                optimized_prompt = optimizer.create_coding_prompt(
                    user_input, context, similar_examples, coding_style
                )
                
                # Compress prompt if too long
                if len(optimized_prompt) > 4000:
                    optimized_prompt = st.session_state.analyzer.compress_prompt(optimized_prompt, 0.7)
                
                # Generate response
                response = model.generate_content(
                    optimized_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                # Store in memory
                memory_key = f"chat_{len(st.session_state.memory.memory_store)}"
                st.session_state.memory.add_memory(
                    memory_key,
                    f"Q: {user_input}\nA: {response.text}",
                    {"type": "chat", "style": coding_style}
                )
                
                # Display response
                st.markdown("### 🤖 Assistant Response:")
                st.markdown(response.text)
                
                # Show RAG context if used
                if use_rag and similar_examples:
                    with st.expander("📚 Retrieved Context"):
                        for i, example in enumerate(similar_examples):
                            st.write(f"**Source {i+1}** (Score: {example['score']:.2f})")
                            st.code(example['content'][:300] + "...")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.header("📁 Codebase Analysis")
    
    if not api_key:
        st.warning("Please configure your Gemini API key in the sidebar.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a zip file containing your codebase",
            type=['zip']
        )
        
        if uploaded_file:
            try:
                # Extract and analyze files
                files = st.session_state.analyzer.extract_files_from_zip(uploaded_file)
                
                st.success(f"Extracted {len(files)} files from codebase")
                
                # File selection
                selected_file = st.selectbox("Select file to analyze:", list(files.keys()))
                
                if selected_file:
                    file_content = files[selected_file]
                    
                    # Show file content
                    st.subheader(f"📄 {selected_file}")
                    st.code(file_content[:1000] + "..." if len(file_content) > 1000 else file_content)
                    
                    analysis_type = st.selectbox(
                        "Analysis Type:",
                        ["general", "performance", "security", "maintainability"]
                    )
                    
                    if st.button("Analyze Code"):
                        # Chunk code if too large
                        chunks = st.session_state.analyzer.chunk_code(file_content, 3000)
                        
                        analysis_results = []
                        for i, chunk in enumerate(chunks):
                            prompt = PromptOptimizer.create_analysis_prompt(chunk, analysis_type)
                            
                            response = model.generate_content(
                                prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=0.3,
                                    max_output_tokens=1500
                                )
                            )
                            
                            analysis_results.append(f"## Chunk {i+1}\n{response.text}")
                        
                        # Store analysis in memory
                        memory_key = f"analysis_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.memory.add_memory(
                            memory_key,
                            "\n\n".join(analysis_results),
                            {"type": "analysis", "file": selected_file, "analysis_type": analysis_type}
                        )
                        
                        # Display results
                        st.markdown("### 🔍 Analysis Results:")
                        for result in analysis_results:
                            st.markdown(result)
                            
            except Exception as e:
                st.error(f"Error processing codebase: {str(e)}")

with tab3:
    st.header("🧠 Memory Management")
    
    # Display current memory
    if st.session_state.memory.memory_store:
        st.subheader("Stored Knowledge")
        
        for key, item in st.session_state.memory.memory_store.items():
            with st.expander(f"📝 {key} ({item['metadata'].get('type', 'unknown')})"):
                st.write(f"**Timestamp:** {item['timestamp']}")
                st.write(f"**Type:** {item['metadata'].get('type', 'N/A')}")
                st.code(item['content'][:500] + "..." if len(item['content']) > 500 else item['content'])
                
                if st.button(f"Delete {key}", key=f"del_{key}"):
                    del st.session_state.memory.memory_store[key]
                    st.session_state.memory._update_vectors()
                    st.experimental_rerun()
    else:
        st.info("No memories stored yet. Use the chat or analysis features to build knowledge.")
    
    st.markdown("---")
    
    # Manual memory addition
    st.subheader("Add Custom Knowledge")
    custom_key = st.text_input("Memory Key:")
    custom_content = st.text_area("Content:", height=100)
    custom_type = st.selectbox("Type:", ["code", "documentation", "example", "note"])
    
    if st.button("Add to Memory") and custom_key and custom_content:
        st.session_state.memory.add_memory(
            custom_key,
            custom_content,
            {"type": custom_type, "source": "manual"}
        )
        st.success("Added to memory!")
        st.experimental_rerun()

with tab4:
    st.header("🔍 Code Review Assistant")
    
    if not api_key:
        st.warning("Please configure your Gemini API key in the sidebar.")
    else:
        review_code = st.text_area("Paste your code for review:", height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            review_focus = st.selectbox(
                "Review Focus:",
                ["Comprehensive", "Performance", "Security", "Best Practices", "Bug Detection"]
            )
        with col2:
            include_fixes = st.checkbox("Include code fixes", value=True)
        
        if st.button("Start Review") and review_code:
            try:
                # Create specialized review prompt
                focus_prompts = {
                    "Comprehensive": "Perform a comprehensive code review covering all aspects.",
                    "Performance": "Focus on performance bottlenecks and optimization opportunities.",
                    "Security": "Identify security vulnerabilities and potential exploits.",
                    "Best Practices": "Evaluate adherence to coding best practices and conventions.",
                    "Bug Detection": "Focus on identifying potential bugs and logical errors."
                }
                
                prompt = f"""As an expert code reviewer, {focus_prompts[review_focus]}

CODE TO REVIEW:
```
{review_code}
```

Please provide:
1. Overall assessment
2. Specific issues found
3. Severity ratings (High/Medium/Low)
4. {'Corrected code examples' if include_fixes else 'Improvement suggestions'}
5. Best practice recommendations

Format your response clearly with sections and examples."""
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=3000
                    )
                )
                
                # Store review in memory
                memory_key = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.memory.add_memory(
                    memory_key,
                    f"ORIGINAL CODE:\n{review_code}\n\nREVIEW:\n{response.text}",
                    {"type": "review", "focus": review_focus}
                )
                
                st.markdown("### 📋 Code Review Results:")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Error during review: {str(e)}")

with tab5:
    st.header("📚 Knowledge Base Search")
    
    search_query = st.text_input("Search your knowledge base:")
    
    if search_query and st.session_state.memory.memory_store:
        results = st.session_state.memory.search_similar(search_query, top_k=5)
        
        if results:
            st.subheader(f"Found {len(results)} relevant results:")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"🔍 Result {i} - {result['key']} (Score: {result['score']:.2f})"):
                    st.write(f"**Type:** {result['metadata'].get('type', 'N/A')}")
                    st.code(result['content'][:800] + "..." if len(result['content']) > 800 else result['content'])
        else:
            st.info("No relevant results found.")
    elif search_query:
        st.info("No knowledge base available. Start using the assistant to build your knowledge base.")

# Footer
st.markdown("---")
st.markdown("""
### 🛠️ Technologies Demonstrated:
- **LLM Deployment**: Gemini Pro integration with optimized parameters
- **RAG System**: Vectorized memory with TF-IDF similarity search
- **Context Optimization**: Intelligent code chunking and prompt compression  
- **Fine-tuning Strategies**: Specialized prompts for different coding tasks
- **Memory Management**: Persistent vectorized storage of conversations and code

*Final Project - AI Internship at Planto.ai (May-July 2025)*
""")
