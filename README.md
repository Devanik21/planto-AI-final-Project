# planto AI final Project

# 🤖 Advanced AI Chatbot Demo

A comprehensive Streamlit application demonstrating multiple AI techniques in a single chatbot interface, built for presentation to stakeholders.

## 🎯 Overview
              
This chatbot demonstrates **Week 3 Advanced AI Techniques** in action:
- **Fine-Tuned Model**: Uses gemma-3n-e4b-it optimized for instruction following
- **RAG (Retrieval Augmented Generation)**: Searches conversation history for relevant context
- **Vectorized Memory**: TF-IDF vectorization for semantic similarity matching
- **Text Compression**: Dynamic prompt optimization and context compression  
- **Large Codebase Fitting**: Intelligent conversation context management

## 🚀 Key Features

- **Unified Interface**: All techniques integrated seamlessly in one chatbot
- **Real-time Transparency**: Shows which techniques are being applied
- **Memory Management**: Persistent vectorized conversation memory
- **Context Optimization**: Automatic compression for large conversations
- **Professional UI**: Clean, presentable interface for stakeholder demos

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API Key
- Basic understanding of AI/ML concepts

## 🛠️ Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files:
   # - app.py
   # - requirements.txt  
   # - README.md
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Configure API Key**
   - Enter your Gemini API key in the sidebar
   - Get your API key from: https://makersuite.google.com/app/apikey

## 💡 How It Works

### 🧠 Fine-Tuned Model
- Uses Gemma-2-9b-it, Google's instruction-tuned language model
- Optimized for conversational AI and following complex instructions

### 📚 RAG Implementation  
- Searches previous conversations for relevant context
- Uses cosine similarity to find most relevant discussions
- Automatically includes relevant context in new responses

### 🔍 Vectorized Memory
- Converts conversations to TF-IDF vectors
- Enables semantic search across conversation history
- Persistent memory that grows with usage

### 📝 Text Compression
- Compresses long contexts while preserving key information
- Intelligent summarization of older conversation parts
- Maintains conversation flow while managing token limits

### 💾 Large Codebase Management
- Handles extensive conversation histories efficiently
- Dynamic context window management
- Balances memory retention with performance

## 🎯 Demo Presentation Points

### For Technical Stakeholders:
1. **Show the sidebar** - Real-time technique indicators
2. **Ask complex questions** - Watch RAG retrieve relevant context
3. **Have a long conversation** - See compression and context management
4. **Check memory stats** - Demonstrate persistent vectorized memory
5. **View technique transparency** - Every response shows what's happening

### Business Value:
- **Reduced API costs** through intelligent context management
- **Better user experience** with relevant conversation memory
- **Scalable architecture** that handles growing conversation data
- **Transparent AI** that shows its reasoning process

## 🔧 Technical Architecture

```
User Input → RAG Retrieval → Context Compression → Fine-tuned Model → Response + Technique Info
     ↓                                                                          ↓
Vectorized Memory ←─────────────────── Memory Update ←─────────────────────────┘
```

## 📊 Metrics & Monitoring

The application tracks:
- Number of vectorized memory entries
- Context compression ratios
- RAG retrieval effectiveness
- Response generation success rates

## 🎨 Customization

### Adding New Techniques:
1. Create new function in `app.py`
2. Add to the processing pipeline in `generate_response()`
3. Update sidebar indicators
4. Include in transparency reporting

### Model Switching:
- Change model name in the `genai.GenerativeModel()` call
- Supported models: `gemma-2-9b-it`, `gemini-1.5-flash`, etc.

## 🚨 Troubleshooting

**API Key Issues:**
- Ensure valid Gemini API key
- Check API quotas and billing

**Memory Performance:**
- Clear memory if response time degrades
- Adjust `max_context_length` for performance tuning

**Installation Problems:**
- Try creating a virtual environment
- Update pip: `pip install --upgrade pip`

## 📈 Performance Notes

- **Memory Usage**: Grows with conversation history
- **API Calls**: One call per user message
- **Response Time**: ~2-5 seconds depending on context size
- **Scalability**: Handles 100+ message conversations efficiently

## 🔮 Future Enhancements

- **Multi-modal support** (images, documents)
- **Advanced vector databases** (Pinecone, Weaviate)
- **Custom fine-tuning** integration
- **Real-time analytics** dashboard
- **Multi-user** conversation memory

## 📞 Support

For technical issues or questions about implementation, refer to:
- Streamlit documentation
- Google Gemini AI documentation  
- Scikit-learn documentation for vector operations

---

**Ready to present!** This demo showcases cutting-edge AI techniques in a user-friendly, transparent interface perfect for stakeholder demonstrations.
- Google Gemini AI documentation  
- Scikit-learn documentation for vector operations

---

**Ready to present!** This demo showcases cutting-edge AI techniques in a user-friendly, transparent interface perfect for stakeholder demonstrations.
