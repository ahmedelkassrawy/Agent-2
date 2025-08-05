from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class RAGModule:
    def __init__(self,
                 model_name: str,
                 embeddings_model: str,
                 doc_path: str):
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self.doc_path = doc_path

        os.environ["GOOGLE_API_KEY"] = "AIzaSyB9o34YFREb_JLj7nXdfNwHfu5Pw9M-Hpw"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_API_KEY"] = ""

        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self):
        try:
            if ".pdf" in self.doc_path:
                loader = PyPDFLoader(self.doc_path)
                docs = loader.load()
                chunks = self.splitter.split_documents(docs)
            else:
                with open(self.doc_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                text_chunks = self.splitter.split_text(text)
                chunks = [Document(page_content=chunk) for chunk in text_chunks]

            return chunks
        except Exception as e:
            raise Exception(f"Error loading documents: {e}")
    
    def create_vector_store(self, chunks):
        vector_store = Chroma.from_documents(
            collection_name="rag_collection",
            documents=chunks,
            embedding=self.hf_embeddings
        )
        
        return vector_store

    def create_chains(self, vector_store):
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Question: {input}

        Context: {context}

        Answer:
        """)

        qa_chain = create_stuff_documents_chain(
            llm=ChatGoogleGenerativeAI(model=self.model_name),
            prompt=prompt
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=qa_chain
        )

        return rag_chain

    def run_rag(self, question: str, rag_chain):
        res = rag_chain.invoke({"input": question})
        return res
    
    def initialize_and_ask(self, question: str):
        """Convenience method to initialize everything and ask a question in one go."""
        try:
            chunks = self.load_documents()
            vector_store = self.create_vector_store(chunks)
            rag_chain = self.create_chains(vector_store)
            result = self.run_rag(question, rag_chain)
            return result
        except Exception as e:
            return {"error": str(e)}