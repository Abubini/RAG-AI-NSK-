from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from typing import List
from app.config import config

class AnswerGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            streaming=False  # Disable streaming for CLI
        )
        
        self.prompt = PromptTemplate.from_template("""
        You are a helpful AI assistant. Answer the question based only on the context below.
        If you don't know the answer, say you don't know. Don't make up information.

        Context:
        {context}

        Question:
        {question}

        Answer in a clear and concise manner:""")
        
        self.chain = (
            RunnableMap({
                "context": lambda x: "\n\n".join([doc.page_content for doc in x["docs"]]),
                "question": lambda x: x["question"]
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_answer(self, question: str, context_docs: List) -> str:
        """Generate answer using retrieved context"""
        if not context_docs:
            return "I couldn't find any relevant information to answer your question."
        
        try:
            result = self.chain.invoke({
                "docs": context_docs,
                "question": question
            })
            return result
        except Exception as e:
            return f"Error generating answer: {str(e)}"