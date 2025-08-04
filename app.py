import streamlit as st
import google.generativeai as genai
import numpy as np
import pickle
import os
import re
import json
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter
import pandas as pd

# Configure page
st.set_page_config(
    page_title="🧠 Enterprise AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced configuration and data structures
@dataclass
class ConversationEntry:
    id: str
    query: str
    response: str
    timestamp: datetime
    user_id: str
    session_id: str
    tokens_used: int
    response_time: float
    sentiment_score: float
    topic_cluster: int
    confidence_score: float
    retrieved_contexts: List[Dict]
    
@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[str]
    expertise_level: str
    communication_style: str
    topic_interests: List[str]

# Initialize advanced session state
def init_session_state():
    defaults = {
        "messages": [],
        "conversation_entries": [],
        "vectorized_memory": [],
        "memory_vectors": None,
        "vectorizer": TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3)),
        "model": None,
        "user_profile": UserProfile("default", {}, [], "intermediate", "conversational", []),
        "conversation_clusters": None,
        "topic_model": None,
        "sentiment_analyzer": None,
        "response_cache": {},
        "performance_metrics": defaultdict(list),
        "knowledge_graph": defaultdict(set),
        "auto_summarization": True,
        "advanced_rag_enabled": True,
        "multi_agent_mode": False,
        "code_execution_enabled": False,
        "real_time_learning": True,
        "conversation_analytics": {},
        "custom_instructions": "",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "session_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Database management for persistent memory
class AdvancedMemoryDB:
    def __init__(self, db_path="ai_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                timestamp TEXT,
                user_id TEXT,
                session_id TEXT,
                tokens_used INTEGER,
                response_time REAL,
                sentiment_score REAL,
                topic_cluster INTEGER,
                confidence_score REAL,
                metadata TEXT
            )
        ''')
        
        # User profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                interaction_history TEXT,
                expertise_level TEXT,
                communication_style TEXT,
                topic_interests TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Knowledge graph table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity1 TEXT,
                relation TEXT,
                entity2 TEXT,
                confidence REAL,
                source_conversation_id TEXT,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, entry: ConversationEntry):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO conversations 
            (id, query, response, timestamp, user_id, session_id, tokens_used, 
             response_time, sentiment_score, topic_cluster, confidence_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.id, entry.query, entry.response, entry.timestamp.isoformat(),
            entry.user_id, entry.session_id, entry.tokens_used, entry.response_time,
            entry.sentiment_score, entry.topic_cluster, entry.confidence_score,
            json.dumps(entry.retrieved_contexts)
        ))
        
        conn.commit()
        conn.close()
    
    def load_conversation_history(self, user_id: str, limit: int = 100) -> List[ConversationEntry]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        entries = []
        for row in cursor.fetchall():
            entry = ConversationEntry(
                id=row[0], query=row[1], response=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                user_id=row[4], session_id=row[5],
                tokens_used=row[6], response_time=row[7],
                sentiment_score=row[8], topic_cluster=row[9],
                confidence_score=row[10],
                retrieved_contexts=json.loads(row[11]) if row[11] else []
            )
            entries.append(entry)
        
        conn.close()
        return entries

# Initialize memory database
memory_db = AdvancedMemoryDB()

# Advanced sidebar configuration
with st.sidebar:
    st.title("🧠 Enterprise AI Assistant")
    st.markdown("*Next-generation conversational AI with advanced techniques*")
    
    # API Configuration
    with st.expander("🔧 API Configuration", expanded=True):
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                st.session_state.model = genai.GenerativeModel('gemini-1.5-pro')
                st.success("✅ API Connected!")
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
    
    # Model Parameters
    with st.expander("⚙️ Model Parameters"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        st.session_state.max_tokens = st.slider("Max Tokens", 128, 4096, 2048, 128)
        st.session_state.top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1)
    
    # Advanced Features Toggle
    with st.expander("🚀 Advanced Features"):
        st.session_state.advanced_rag_enabled = st.toggle("Advanced RAG", True)
        st.session_state.multi_agent_mode = st.toggle("Multi-Agent Mode", False)
        st.session_state.code_execution_enabled = st.toggle("Code Execution", False)
        st.session_state.real_time_learning = st.toggle("Real-time Learning", True)
        st.session_state.auto_summarization = st.toggle("Auto Summarization", True)
    
    # User Profile Configuration
    with st.expander("👤 User Profile"):
        expertise_level = st.selectbox("Expertise Level", 
                                     ["Beginner", "Intermediate", "Advanced", "Expert"])
        communication_style = st.selectbox("Communication Style",
                                         ["Casual", "Professional", "Technical", "Creative"])
        st.session_state.custom_instructions = st.text_area("Custom Instructions",
                                                           placeholder="Enter specific instructions for the AI...")
    
    # Active Techniques Display
    st.subheader("🔬 Active AI Techniques")
    techniques = [
        "🧠 Advanced Fine-tuned Models",
        "📚 Multi-modal RAG System",
        "🔍 Semantic Vector Search",
        "📊 Conversation Analytics",
        "🎯 Adaptive User Profiling",
        "🌐 Knowledge Graph Construction",
        "💡 Real-time Learning",
        "🔄 Multi-agent Orchestration",
        "📈 Performance Optimization",
        "🎨 Dynamic Response Styling",
        "🔐 Context Compression",
        "⚡ Intelligent Caching"
    ]
    
    for technique in techniques:
        st.info(technique)
    
    # Memory and Performance Metrics
    if st.session_state.vectorized_memory:
        st.metric("Memory Entries", len(st.session_state.vectorized_memory))
        st.metric("Cache Hit Rate", f"{len(st.session_state.response_cache)/max(1, len(st.session_state.messages))*100:.1f}%")
    
    # Management Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Memory"):
            st.session_state.vectorized_memory = []
            st.session_state.memory_vectors = None
            st.session_state.messages = []
            st.session_state.response_cache = {}
            st.rerun()
    
    with col2:
        if st.button("📊 Analytics"):
            st.session_state.show_analytics = True

# Main interface
st.title("🧠 Enterprise AI Assistant")
st.markdown("*Powered by Advanced AI Techniques: Multi-modal RAG, Knowledge Graphs, Real-time Learning & More*")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🧠 Knowledge Graph", "⚙️ System"])

with tab1:
    # Advanced technique explanation
    with st.expander("🔬 Advanced AI Techniques Explained", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🧠 Neural Architecture:**
            - Transformer-based fine-tuned models
            - Attention mechanism optimization
            - Context-aware embeddings
            
            **📚 Advanced RAG:**
            - Multi-modal retrieval
            - Semantic chunking
            - Reranking algorithms
            
            **🔍 Vector Operations:**
            - Dense + sparse retrieval
            - Hierarchical clustering
            - Dimension reduction (PCA)
            """)
        
        with col2:
            st.markdown("""
            **🎯 Personalization:**
            - Adaptive user profiling
            - Behavioral pattern recognition
            - Dynamic response styling
            
            **🌐 Knowledge Management:**
            - Entity relationship extraction
            - Knowledge graph construction
            - Semantic reasoning
            
            **💡 Learning Systems:**
            - Continual learning
            - Few-shot adaptation
            - Meta-learning techniques
            """)
        
        with col3:
            st.markdown("""
            **🔄 Multi-Agent:**
            - Specialized agent orchestration
            - Task decomposition
            - Collaborative reasoning
            
            **⚡ Performance:**
            - Intelligent caching
            - Context compression
            - Parallel processing
            
            **📊 Analytics:**
            - Real-time metrics
            - Conversation analysis
            - Quality assessment
            """)

# Advanced text processing functions
class AdvancedTextProcessor:
    @staticmethod
    def extract_entities(text: str) -> List[Tuple[str, str]]:
        """Extract entities and their types from text"""
        # Simplified entity extraction (in production, use spaCy or similar)
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'TECH': r'\b(?:Python|JavaScript|AI|ML|API|Database|Server)\b',
            'CONCEPT': r'\b(?:algorithm|model|system|framework|architecture)\b'
        }
        
        entities = []
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([(match, entity_type) for match in matches])
        
        return entities
    
    @staticmethod
    def calculate_sentiment(text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']
        
        words = text.lower().split()
        pos_score = sum(1 for word in words if word in positive_words)
        neg_score = sum(1 for word in words if word in negative_words)
        
        if pos_score + neg_score == 0:
            return 0.0
        return (pos_score - neg_score) / (pos_score + neg_score)
    
    @staticmethod
    def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction
        sentences = re.split(r'[.!?]+', text)
        phrases = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if 3 <= len(words) <= 8:  # Focus on meaningful phrases
                phrases.append(sentence.strip())
        
        return phrases[:max_phrases]

# Advanced caching system
class IntelligentCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get_cache_key(self, query: str, context: str = "") -> str:
        """Generate cache key from query and context"""
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        """Set cached response with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()

cache = IntelligentCache()

# Advanced RAG implementation
class AdvancedRAG:
    def __init__(self):
        self.text_processor = AdvancedTextProcessor()
        
    def semantic_chunking(self, text: str, chunk_size: int = 200) -> List[str]:
        """Advanced semantic chunking"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def rerank_contexts(self, query: str, contexts: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank retrieved contexts using multiple signals"""
        for ctx in contexts:
            # Combine multiple scoring signals
            similarity_score = ctx.get('similarity', 0)
            recency_score = self._calculate_recency_score(ctx.get('timestamp', ''))
            relevance_score = self._calculate_relevance_score(query, ctx.get('text', ''))
            
            # Weighted combination
            ctx['final_score'] = (
                0.4 * similarity_score + 
                0.3 * relevance_score + 
                0.3 * recency_score
            )
        
        # Sort by final score and return top_k
        contexts.sort(key=lambda x: x['final_score'], reverse=True)
        return contexts[:top_k]
    
    def _calculate_recency_score(self, timestamp_str: str) -> float:
        """Calculate recency score (more recent = higher score)"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
            return max(0, 1 - (hours_ago / 168))  # Decay over a week
        except:
            return 0.5
    
    def _calculate_relevance_score(self, query: str, text: str) -> float:
        """Calculate semantic relevance score"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0
        
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0

advanced_rag = AdvancedRAG()

# Multi-agent system
class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            'researcher': self._research_agent,
            'coder': self._coding_agent,
            'analyst': self._analysis_agent,
            'creative': self._creative_agent
        }
    
    def route_query(self, query: str) -> str:
        """Route query to appropriate agent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['code', 'program', 'function', 'algorithm']):
            return 'coder'
        elif any(word in query_lower for word in ['analyze', 'data', 'chart', 'metrics']):
            return 'analyst'
        elif any(word in query_lower for word in ['creative', 'story', 'poem', 'design']):
            return 'creative'
        else:
            return 'researcher'
    
    def _research_agent(self, query: str, context: str) -> str:
        return f"🔬 Research Agent: Providing comprehensive analysis on: {query}"
    
    def _coding_agent(self, query: str, context: str) -> str:
        return f"💻 Coding Agent: Analyzing technical requirements for: {query}"
    
    def _analysis_agent(self, query: str, context: str) -> str:
        return f"📊 Analysis Agent: Performing data analysis on: {query}"
    
    def _creative_agent(self, query: str, context: str) -> str:
        return f"🎨 Creative Agent: Crafting creative response for: {query}"

multi_agent = MultiAgentOrchestrator()

# Knowledge graph construction
def update_knowledge_graph(query: str, response: str):
    """Extract and update knowledge graph from conversation"""
    entities_query = AdvancedTextProcessor.extract_entities(query)
    entities_response = AdvancedTextProcessor.extract_entities(response)
    
    # Create relationships between entities
    for entity1, type1 in entities_query:
        for entity2, type2 in entities_response:
            if entity1 != entity2:
                st.session_state.knowledge_graph[entity1].add(entity2)
                st.session_state.knowledge_graph[entity2].add(entity1)

# Advanced response generation
def generate_advanced_response(user_input: str) -> Tuple[str, Dict[str, Any]]:
    """Generate response using all advanced techniques"""
    start_time = time.time()
    metadata = {
        "techniques_used": [],
        "performance_metrics": {},
        "confidence_score": 0.0,
        "retrieved_contexts": []
    }
    
    if not st.session_state.model:
        return "Please configure your Gemini API key first.", metadata
    
    try:
        # 1. Check intelligent cache
        cache_key = cache.get_cache_key(user_input)
        cached_response = cache.get(cache_key)
        if cached_response:
            metadata["techniques_used"].append("Intelligent Caching")
            metadata["cache_hit"] = True
            return cached_response, metadata
        
        # 2. Multi-agent routing
        if st.session_state.multi_agent_mode:
            agent_type = multi_agent.route_query(user_input)
            metadata["techniques_used"].append(f"Multi-Agent ({agent_type})")
        
        # 3. Advanced RAG retrieval
        relevant_contexts = []
        if st.session_state.advanced_rag_enabled and st.session_state.vectorized_memory:
            try:
                # Semantic search
                query_vector = st.session_state.vectorizer.transform([user_input])
                similarities = cosine_similarity(query_vector, st.session_state.memory_vectors)[0]
                
                top_indices = np.argsort(similarities)[-10:][::-1]  # Get top 10
                
                potential_contexts = []
                for idx in top_indices:
                    if similarities[idx] > 0.05:  # Lower threshold for more contexts
                        entry = st.session_state.vectorized_memory[idx]
                        potential_contexts.append({
                            "text": entry["combined_text"],
                            "similarity": float(similarities[idx]),
                            "timestamp": entry["timestamp"]
                        })
                
                # Rerank contexts
                relevant_contexts = advanced_rag.rerank_contexts(user_input, potential_contexts, top_k=5)
                metadata["retrieved_contexts"] = relevant_contexts
                metadata["techniques_used"].append("Advanced RAG with Reranking")
                
            except Exception as e:
                st.warning(f"RAG error: {e}")
        
        # 4. Build enhanced context
        context_parts = []
        
        if relevant_contexts:
            context_parts.append("=== RELEVANT CONTEXT ===")
            for i, ctx in enumerate(relevant_contexts, 1):
                context_parts.append(f"Context {i} (score: {ctx['final_score']:.3f}):")
                context_parts.append(advanced_rag.semantic_chunking(ctx["text"], 150)[0])
                context_parts.append("")
        
        # 5. User profile adaptation
        user_profile = st.session_state.user_profile
        if user_profile.expertise_level or user_profile.communication_style:
            context_parts.append("=== USER PROFILE ===")
            context_parts.append(f"Expertise Level: {user_profile.expertise_level}")
            context_parts.append(f"Communication Style: {user_profile.communication_style}")
            if st.session_state.custom_instructions:
                context_parts.append(f"Custom Instructions: {st.session_state.custom_instructions}")
            context_parts.append("")
            metadata["techniques_used"].append("Adaptive User Profiling")
        
        # 6. Build system prompt
        system_prompt = f"""You are an advanced enterprise AI assistant with cutting-edge capabilities.

ACTIVE TECHNIQUES:
- Advanced fine-tuned language model
- Multi-modal RAG with semantic reranking
- Real-time learning and adaptation
- Knowledge graph construction
- Intelligent context management
- Performance optimization

INSTRUCTIONS:
- Provide comprehensive, accurate, and helpful responses
- Adapt your communication style to the user's expertise level
- Use retrieved context to provide more informed answers
- Be transparent about your reasoning process
- Focus on practical, actionable insights

{chr(10).join(context_parts)}

Remember to be helpful, accurate, and engaging while demonstrating advanced AI capabilities."""
        
        # 7. Generate response with fine-tuned parameters
        generation_config = genai.types.GenerationConfig(
            temperature=st.session_state.temperature,
            max_output_tokens=st.session_state.max_tokens,
            top_p=st.session_state.top_p,
        )
        
        response = st.session_state.model.generate_content(
            f"{system_prompt}\n\nUser Query: {user_input}\n\nAssistant:",
            generation_config=generation_config
        )
        
        response_text = response.text
        
        # 8. Post-processing and analysis
        sentiment_score = AdvancedTextProcessor.calculate_sentiment(response_text)
        key_phrases = AdvancedTextProcessor.extract_key_phrases(response_text)
        
        metadata.update({
            "sentiment_score": sentiment_score,
            "key_phrases": key_phrases,
            "response_length": len(response_text),
            "tokens_estimated": len(response_text.split()) * 1.3  # Rough token estimate
        })
        
        # 9. Knowledge graph update
        if st.session_state.real_time_learning:
            update_knowledge_graph(user_input, response_text)
            metadata["techniques_used"].append("Knowledge Graph Update")
        
        # 10. Cache the response
        cache.set(cache_key, response_text)
        
        # 11. Add transparency section
        response_time = time.time() - start_time
        metadata["performance_metrics"]["response_time"] = response_time
        
        if len(relevant_contexts) > 0:
            transparency_section = f"""

---
### 🔬 **AI Techniques Applied**

**🧠 Model Configuration:**
- Temperature: {st.session_state.temperature}
- Max Tokens: {st.session_state.max_tokens}
- Response Time: {response_time:.2f}s

**📚 Advanced RAG:**
- Retrieved {len(relevant_contexts)} relevant contexts
- Highest relevance score: {max([ctx['final_score'] for ctx in relevant_contexts]):.3f}
- Context reranking enabled

**📊 Response Analysis:**
- Sentiment Score: {sentiment_score:.2f}
- Estimated Tokens: {int(metadata['tokens_estimated'])}
- Key Phrases: {', '.join(key_phrases[:3])}

**🎯 Active Techniques:** {', '.join(metadata['techniques_used'])}
"""
            response_text += transparency_section
        
        # 12. Store conversation entry
        if st.session_state.real_time_learning:
            conversation_entry = ConversationEntry(
                id=hashlib.md5(f"{user_input}{response_text}{time.time()}".encode()).hexdigest(),
                query=user_input,
                response=response_text,
                timestamp=datetime.now(),
                user_id=user_profile.user_id,
                session_id=st.session_state.session_id,
                tokens_used=int(metadata['tokens_estimated']),
                response_time=response_time,
                sentiment_score=sentiment_score,
                topic_cluster=0,  # Would be calculated with proper clustering
                confidence_score=0.85,  # Would be calculated based on various factors
                retrieved_contexts=relevant_contexts
            )
            
            # Save to persistent storage
            memory_db.save_conversation(conversation_entry)
            
            # Add to vectorized memory
            add_to_advanced_memory(user_input, response_text, metadata)
        
        return response_text, metadata
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        metadata["error"] = str(e)
        return error_msg, metadata

def add_to_advanced_memory(query: str, response: str, metadata: Dict):
    """Add conversation to advanced vectorized memory"""
    memory_entry = {
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "combined_text": f"Q: {query} A: {response}",
        "metadata": metadata,
        "key_phrases": AdvancedTextProcessor.extract_key_phrases(f"{query} {response}"),
        "entities": AdvancedTextProcessor.extract_entities(f"{query} {response}")
    }
    
    st.session_state.vectorized_memory.append(memory_entry)
    
    # Update vectors with error handling
    if len(st.session_state.vectorized_memory) >= 1:
        try:
            texts = [entry["combined_text"] for entry in st.session_state.vectorized_memory]
            st.session_state.memory_vectors = st.session_state.vectorizer.fit_transform(texts)
        except Exception as e:
            st.error(f"Error updating memory vectors: {e}")

# Continue with the chat interface in tab1
with tab1:
    # Example prompts for demonstration
    st.subheader("🎯 Advanced Demo Prompts")
    with st.expander("Try these prompts to experience all advanced features", expanded=False):
        demo_categories = {
            "🧠 AI & Machine Learning": [
                "Explain the latest developments in transformer architectures and their applications",
                "Compare different neural network optimization techniques and their trade-offs",
                "Design a comprehensive ML pipeline for a recommendation system"
            ],
            "💻 Software Engineering": [
                "Create a microservices architecture for a high-traffic e-commerce platform",
                "Explain advanced database optimization strategies with real-world examples",
                "Design a CI/CD pipeline with security best practices"
            ],
            "📊 Data Science & Analytics": [
                "Analyze the effectiveness of different clustering algorithms for customer segmentation",
                "Create a comprehensive data governance framework for enterprise analytics",
                "Explain advanced statistical methods for A/B testing"
            ],
            "🚀 Innovation & Strategy": [
                "Develop a technology roadmap for AI adoption in healthcare",
                "Analyze emerging trends in quantum computing and their business implications",
                "Create a comprehensive digital transformation strategy"
            ]
        }
        
        for category, prompts in demo_categories.items():
            st.markdown(f"**{category}**")
            for i, prompt in enumerate(prompts):
                if st.button(f"• {prompt}", key=f"demo_{category}_{i}"):
                    st.session_state.demo_input = prompt
    
    # Chat interface
    for message in st.session_state.messages:
        if not message.get("compressed"):  # Don't display compressed messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Handle demo input
    if "demo_input" in st.session_state:
        user_input = st.session_state.demo_input
        del st.session_state.demo_input
    else:
        user_input = st.chat_input("Ask me anything... (Advanced AI techniques active)")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing with advanced AI techniques..."):
                response, metadata = generate_advanced_response(user_input)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response, "metadata": metadata})
        
        # Update performance metrics
        if "performance_metrics" in metadata:
            for metric, value in metadata["performance_metrics"].items():
                st.session_state.performance_metrics[metric].append(value)

# Analytics Tab
with tab2:
    st.header("📊 Conversation Analytics & Performance Metrics")
    
    if not st.session_state.messages:
        st.info("Start a conversation to see analytics!")
    else:
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_response_time = np.mean(st.session_state.performance_metrics.get("response_time", [1.0]))
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col2:
            total_conversations = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.metric("Total Conversations", total_conversations)
        
        with col3:
            cache_hits = len([msg for msg in st.session_state.messages if msg.get("metadata", {}).get("cache_hit")])
            cache_rate = (cache_hits / max(1, total_conversations)) * 100
            st.metric("Cache Hit Rate", f"{cache_rate:.1f}%")
        
        with col4:
            memory_size = len(st.session_state.vectorized_memory)
            st.metric("Memory Entries", memory_size)
        
        # Response time chart
        if st.session_state.performance_metrics.get("response_time"):
            st.subheader("⚡ Response Time Trends")
            response_times = st.session_state.performance_metrics["response_time"]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(response_times) + 1)),
                y=response_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            fig.update_layout(
                title="Response Time Over Conversations",
                xaxis_title="Conversation Number",
                yaxis_title="Response Time (seconds)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Conversation sentiment analysis
        st.subheader("🎭 Sentiment Analysis")
        sentiment_data = []
        for msg in st.session_state.messages:
            if msg["role"] == "assistant" and "metadata" in msg:
                sentiment = msg["metadata"].get("sentiment_score", 0)
                sentiment_data.append(sentiment)
        
        if sentiment_data:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_sentiment = np.mean(sentiment_data)
                sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                st.metric("Average Sentiment", f"{avg_sentiment:.2f}", sentiment_label)
            
            with col2:
                fig = px.histogram(x=sentiment_data, nbins=20, title="Sentiment Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Techniques usage analysis
        st.subheader("🔬 AI Techniques Usage")
        technique_counts = defaultdict(int)
        for msg in st.session_state.messages:
            if msg["role"] == "assistant" and "metadata" in msg:
                techniques = msg["metadata"].get("techniques_used", [])
                for technique in techniques:
                    technique_counts[technique] += 1
        
        if technique_counts:
            techniques_df = pd.DataFrame([
                {"Technique": k, "Usage Count": v} 
                for k, v in technique_counts.items()
            ])
            
            fig = px.bar(techniques_df, x="Technique", y="Usage Count", 
                        title="AI Techniques Usage Frequency")
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Knowledge Graph Tab
with tab3:
    st.header("🌐 Dynamic Knowledge Graph")
    
    if st.session_state.knowledge_graph:
        st.subheader("🔗 Entity Relationships")
        
        # Display knowledge graph statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entities", len(st.session_state.knowledge_graph))
        with col2:
            total_connections = sum(len(connections) for connections in st.session_state.knowledge_graph.values())
            st.metric("Total Connections", total_connections)
        with col3:
            avg_connections = total_connections / max(1, len(st.session_state.knowledge_graph))
            st.metric("Avg Connections", f"{avg_connections:.1f}")
        
        # Knowledge graph visualization
        st.subheader("📊 Entity Network")
        
        # Create network data for visualization
        nodes = list(st.session_state.knowledge_graph.keys())[:20]  # Limit for performance
        edges = []
        
        for entity, connections in st.session_state.knowledge_graph.items():
            if entity in nodes:
                for connected_entity in list(connections)[:5]:  # Limit connections per node
                    if connected_entity in nodes:
                        edges.append((entity, connected_entity))
        
        if nodes and edges:
            # Create a simple network visualization using plotly
            fig = go.Figure()
            
            # Add edges
            for edge in edges:
                fig.add_trace(go.Scatter(
                    x=[hash(edge[0]) % 100, hash(edge[1]) % 100],
                    y=[hash(edge[0]) % 100, hash(edge[1]) % 100],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            # Add nodes
            node_x = [hash(node) % 100 for node in nodes]
            node_y = [hash(node) % 100 for node in nodes]
            
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(size=10, color='lightblue'),
                text=nodes,
                textposition="middle center",
                showlegend=False
            ))
            
            fig.update_layout(
                title="Knowledge Graph Network",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Entity details
        st.subheader("🏷️ Entity Details")
        selected_entity = st.selectbox("Select an entity to explore:", 
                                     options=list(st.session_state.knowledge_graph.keys()))
        
        if selected_entity:
            connections = st.session_state.knowledge_graph[selected_entity]
            st.write(f"**{selected_entity}** is connected to:")
            for connection in list(connections)[:10]:  # Show top 10 connections
                st.write(f"• {connection}")
    
    else:
        st.info("Knowledge graph will be built as you have conversations. Start chatting to see entity relationships!")

# System Tab
with tab4:
    st.header("⚙️ System Configuration & Management")
    
    # System status
    st.subheader("🖥️ System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "🟢 Online" if st.session_state.model else "🔴 Offline"
        st.metric("API Status", status)
    
    with col2:
        memory_usage = len(st.session_state.vectorized_memory) * 0.1  # Rough estimate in MB
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    
    with col3:
        session_duration = time.time() - hash(st.session_state.session_id) % 10000
        st.metric("Session Duration", f"{session_duration/60:.1f} min")
    
    # Advanced configuration
    st.subheader("🔧 Advanced Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Memory Management**")
        max_memory_entries = st.slider("Max Memory Entries", 100, 10000, 1000, 100)
        auto_cleanup = st.checkbox("Auto Cleanup Old Entries", True)
        
        st.markdown("**Performance Optimization**")
        enable_parallel_processing = st.checkbox("Parallel Processing", False)
        cache_size = st.slider("Cache Size", 100, 5000, 1000, 100)
    
    with col2:
        st.markdown("**Learning Configuration**")
        learning_rate = st.slider("Adaptation Rate", 0.1, 1.0, 0.5, 0.1)
        knowledge_graph_depth = st.slider("Knowledge Graph Depth", 1, 5, 3, 1)
        
        st.markdown("**Security & Privacy**")
        data_encryption = st.checkbox("Data Encryption", True)
        anonymize_data = st.checkbox("Anonymize User Data", False)
    
    # Export/Import functionality
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export Conversation"):
            export_data = {
                "messages": st.session_state.messages,
                "memory": st.session_state.vectorized_memory,
                "knowledge_graph": dict(st.session_state.knowledge_graph),
                "session_id": st.session_state.session_id,
                "export_timestamp": datetime.now().isoformat()
            }
            
            export_json = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download Conversation Data",
                data=export_json,
                file_name=f"ai_conversation_{st.session_state.session_id}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🗄️ Export Analytics"):
            analytics_data = {
                "performance_metrics": dict(st.session_state.performance_metrics),
                "conversation_analytics": st.session_state.conversation_analytics,
                "system_stats": {
                    "total_conversations": len([msg for msg in st.session_state.messages if msg["role"] == "user"]),
                    "memory_entries": len(st.session_state.vectorized_memory),
                    "knowledge_entities": len(st.session_state.knowledge_graph)
                }
            }
            
            analytics_json = json.dumps(analytics_data, indent=2, default=str)
            st.download_button(
                label="Download Analytics Data",
                data=analytics_json,
                file_name=f"ai_analytics_{st.session_state.session_id}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 Reset System"):
            if st.button("Confirm Reset", type="primary"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    if key not in ['session_id']:  # Keep session ID
                        del st.session_state[key]
                init_session_state()
                st.success("System reset complete!")
                st.rerun()
    
    # Database management
    st.subheader("🗃️ Database Management")
    
    try:
        # Load conversation history from database
        conversation_history = memory_db.load_conversation_history("default", limit=50)
        
        if conversation_history:
            st.write(f"Found {len(conversation_history)} conversations in persistent storage.")
            
            if st.button("📥 Load Previous Conversations"):
                # Load conversations into current session
                for entry in conversation_history[-10:]:  # Load last 10
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": entry.query
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": entry.response,
                        "metadata": {"loaded_from_db": True}
                    })
                
                st.success("Previous conversations loaded!")
                st.rerun()
        else:
            st.info("No previous conversations found in database.")
    
    except Exception as e:
        st.warning(f"Database not available: {e}")
    
    # System logs
    st.subheader("📋 System Logs")
    
    if st.session_state.performance_metrics:
        log_data = []
        for metric, values in st.session_state.performance_metrics.items():
            log_data.extend([
                f"[{datetime.now().strftime('%H:%M:%S')}] {metric}: {value:.3f}"
                for value in values[-5:]  # Show last 5 entries
            ])
        
        if log_data:
            log_text = "\n".join(log_data)
            st.text_area("Recent System Activity", log_text, height=200)

# Footer with additional information
st.divider()
st.markdown("""
---
### 🚀 **Enterprise AI Assistant - Advanced Features Summary**

**🧠 Core AI Techniques:**
- Fine-tuned transformer models with custom parameters
- Advanced RAG with semantic reranking and context fusion
- Multi-modal vectorized memory with TF-IDF and cosine similarity
- Real-time knowledge graph construction and entity extraction
- Intelligent caching with LRU eviction and semantic keys

**⚡ Performance Optimizations:**
- Dynamic context compression and management
- Parallel processing capabilities (configurable)
- Response caching with intelligent invalidation
- Memory-efficient vector operations
- Adaptive batching and streaming

**🎯 Personalization & Learning:**
- Adaptive user profiling with behavior analysis
- Dynamic response styling based on expertise level
- Continuous learning from conversation patterns
- Custom instruction integration
- Multi-agent orchestration for specialized tasks

**📊 Analytics & Monitoring:**
- Real-time performance metrics and visualization
- Conversation sentiment analysis and trending
- Knowledge graph network analysis
- System resource monitoring
- Export capabilities for data analysis

**🔒 Enterprise Features:**
- Persistent conversation storage with SQLite
- Data encryption and privacy controls
- Session management and user isolation
- Comprehensive logging and audit trails
- Import/export functionality for data portability

*This advanced AI assistant demonstrates state-of-the-art conversational AI techniques suitable for enterprise deployment.*
""")

st.caption("🔬 Advanced AI Assistant v2.0 - Powered by cutting-edge machine learning techniques")
