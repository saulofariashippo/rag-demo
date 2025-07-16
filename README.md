# Simple RAG Demo 🚀

A **Retrieval-Augmented Generation (RAG)** system that combines semantic search with AI text generation to answer questions based on your knowledge base.

## 🎯 What is RAG?

RAG combines two powerful AI techniques:

1. **🔍 Retrieval**: Find the most relevant documents for a question
2. **🤖 Generation**: Generate answers based on the retrieved context

This creates AI systems that can answer questions accurately using your specific documents/data, rather than just relying on what the model learned during training.

## ✨ Features

- **Simple & Fast**: No complex dependencies or massive downloads
- **Semantic Search**: Uses sentence transformers for meaning-based document retrieval
- **Smart Answers**: Generates contextual answers using Google's FLAN-T5 model
- **Confidence Scores**: Shows how relevant each retrieved document is
- **Easy to Extend**: Add your own documents and knowledge base
- **Cross-Platform**: Works on macOS, Linux, and Windows

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd rag
python3 -m venv rag-demo
source rag-demo/bin/activate  # On Windows: rag-demo\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python rag_demo.py
```

### 3. Expected Output

```
🚀 Starting Simple RAG Demo
==================================================
🔄 Loading embedding model...
🔄 Loading text generation model...
🔄 Creating document embeddings...
✅ RAG system ready!

❓ Question: Who won the Women's World Cup in 2023?

🔍 Retrieving relevant documents...
  📄 Rank 1 (score: 0.760): Spain won the 2023 FIFA Women's World Cup...
  📄 Rank 2 (score: 0.742): The 2023 tournament featured 32 teams...
  📄 Rank 3 (score: 0.692): The 2023 FIFA Women's World Cup was held...

🤖 Generating answer...
✅ Answer: Spain
```

## 🛠️ How It Works

### Architecture Overview

```
Question → [Embedding] → [Similarity Search] → [Top Documents] → [LLM] → Answer
```

### Step-by-Step Process

1. **📝 Document Preprocessing**: Convert documents to vector embeddings
2. **❓ Question Processing**: Convert user question to vector embedding
3. **🔍 Retrieval**: Find most similar documents using cosine similarity
4. **📋 Context Creation**: Combine top documents into context
5. **🤖 Generation**: Feed question + context to language model
6. **✅ Answer**: Return generated response

### Key Components

| Component      | Model                  | Purpose                          |
| -------------- | ---------------------- | -------------------------------- |
| **Embeddings** | `all-MiniLM-L6-v2`     | Convert text to semantic vectors |
| **Generation** | `google/flan-t5-small` | Generate answers from context    |
| **Similarity** | Cosine Similarity      | Find most relevant documents     |

## 📊 Performance

- **First Run**: ~30 seconds (downloads models)
- **Subsequent Runs**: ~5 seconds (models cached)
- **Memory Usage**: ~1GB RAM
- **Model Sizes**:
  - Embedding model: 22MB
  - Generation model: 308MB

## 🔧 Customization

### Add Your Own Documents

Replace the `documents` list in `rag_demo.py`:

```python
documents = [
    "Your first document here...",
    "Your second document here...",
    "Add as many as you need..."
]
```

### Change Models

```python
# For better embeddings (larger but more accurate)
self.encoder = SentenceTransformer('all-mpnet-base-v2')

# For better generation (larger but more capable)
self.generator = pipeline("text2text-generation", model="google/flan-t5-base")
```

### Adjust Retrieval

```python
# Get more/fewer documents per question
retrieved_docs = self.retrieve(question, top_k=5)  # Default: 3

# Longer answers
result = self.generator(prompt, max_length=200)  # Default: 150
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: `OMP: Error #15: Initializing libomp.dylib`  
**Solution**: The code automatically sets `KMP_DUPLICATE_LIB_OK=TRUE`

**Problem**: `ModuleNotFoundError: No module named 'sentence_transformers'`  
**Solution**: Make sure you activated the virtual environment and installed requirements

**Problem**: Models downloading slowly  
**Solution**: This only happens on first run. Models are cached for future use.

### System Requirements

- **Python**: 3.8+ (tested on 3.13)
- **RAM**: 2GB+ recommended
- **Storage**: 500MB for models
- **Internet**: Required for first run (model downloads)

## 📁 Project Structure

```
rag/
├── rag_demo.py          # Main RAG implementation
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── rag-demo/           # Virtual environment
```

## 🔬 Technical Details

### Dependencies

- **sentence-transformers**: Semantic embeddings
- **transformers**: Hugging Face model hub
- **torch**: PyTorch backend
- **scikit-learn**: Cosine similarity computation
- **numpy**: Numerical operations

### Model Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

  - 384-dimensional embeddings
  - Trained on 1B+ sentence pairs
  - Best balance of speed vs accuracy

- **Generation Model**: `google/flan-t5-small`
  - 80M parameters
  - Instruction-tuned for Q&A tasks
  - Fast inference on CPU

## 🚀 Next Steps

### Scale Up

- Add vector database (Pinecone, Weaviate, Chroma)
- Use larger/better models
- Implement document chunking for long texts

### Production Features

- Add caching layer
- Implement batch processing
- Add API endpoints
- Monitor performance metrics

### Advanced RAG

- Multi-query retrieval
- Re-ranking models
- Citation tracking
- Conversation memory

## 📜 License

MIT License - feel free to use this code for learning, prototyping, and production!

## 🤝 Contributing

Found a bug or want to add a feature? Feel free to open an issue or submit a pull request!

---

**Happy RAG-ing!** 🎉
