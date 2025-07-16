import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict on macOS

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents to retrieve from (in a real app, this would be your knowledge base)
documents = [
    "The 2023 FIFA Women's World Cup was held in Australia and New Zealand from 20 July to 20 August 2023.",
    "Spain won the 2023 FIFA Women's World Cup, defeating England 1-0 in the final on 20 August 2023.",
    "The final was played at Stadium Australia in Sydney, with Olga Carmona scoring the winning goal for Spain.",
    "England reached their first Women's World Cup final but were defeated by Spain in Sydney.",
    "The 2023 tournament featured 32 teams for the first time in Women's World Cup history.",
    "Australia and New Zealand co-hosted the tournament, making it the first Women's World Cup held in the Southern Hemisphere.",
    "Spain's victory marked their first-ever FIFA Women's World Cup title.",
    "The tournament saw record attendance figures and global viewership for women's football.",
]

class SimpleRAG:
    def __init__(self, documents):
        print("🔄 Loading embedding model...")
        # Use a lightweight, reliable embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("🔄 Loading text generation model...")
        # Use a simple text generation pipeline
        self.generator = pipeline(
            "text2text-generation", 
            model="google/flan-t5-small",
            max_length=100,
            do_sample=False
        )
        
        # Store documents and create embeddings
        self.documents = documents
        print("🔄 Creating document embeddings...")
        self.doc_embeddings = self.encoder.encode(documents)
        
        print("✅ RAG system ready!")
    
    def retrieve(self, question, top_k=3):
        """Retrieve the most relevant documents for a question"""
        # Encode the question
        question_embedding = self.encoder.encode([question])
        
        # Calculate cosine similarity between question and all documents
        similarities = cosine_similarity(question_embedding, self.doc_embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return retrieved documents with scores
        retrieved_docs = []
        for i, idx in enumerate(top_indices):
            retrieved_docs.append({
                'document': self.documents[idx],
                'score': float(similarities[idx]),
                'rank': i + 1
            })
        
        return retrieved_docs
    
    def generate_answer(self, question, retrieved_docs):
        """Generate an answer based on the question and retrieved documents"""
        # Create context from retrieved documents
        context = "\n".join([doc['document'] for doc in retrieved_docs])
        
        # Create a prompt for the generator
        prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer based on the context:"
        
        # Generate answer
        result = self.generator(prompt, max_length=150, do_sample=False)
        return result[0]['generated_text']
    
    def answer(self, question):
        """Full RAG pipeline: retrieve + generate"""
        print(f"\n❓ Question: {question}")
        
        # Step 1: Retrieve relevant documents
        print("\n🔍 Retrieving relevant documents...")
        retrieved_docs = self.retrieve(question, top_k=3)
        
        for doc in retrieved_docs:
            print(f"  📄 Rank {doc['rank']} (score: {doc['score']:.3f}): {doc['document'][:100]}...")
        
        # Step 2: Generate answer
        print("\n🤖 Generating answer...")
        answer = self.generate_answer(question, retrieved_docs)
        
        print(f"\n✅ Answer: {answer}")
        return answer, retrieved_docs

# Initialize and test the RAG system
if __name__ == "__main__":
    print("🚀 Starting Simple RAG Demo")
    print("=" * 50)
    
    # Create RAG system
    rag = SimpleRAG(documents)
    
    # Test questions
    questions = [
        "Who won the Women's World Cup in 2023?",
        "Where was the 2023 Women's World Cup final played?",
        "Which countries hosted the 2023 Women's World Cup?",
        "How many teams participated in the 2023 tournament?"
    ]
    
    for question in questions:
        answer, docs = rag.answer(question)
        print("\n" + "="*50)
