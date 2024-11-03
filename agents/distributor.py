from typing import Annotated, Dict, TypedDict, List, Tuple
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt.tool_executor import ToolExecutor
import operator
from typing import Any, Sequence
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from tavily import TavilyClient
import os
from collections import deque

class ExportChatAgent:
    def __init__(self, pdf_path: str, tavily_api_key: str):
        self.chat_history = deque(maxlen=3)  # Keep only last 3 Q&A pairs
        self.setup_components(pdf_path, tavily_api_key)
        self.create_chains()
        self.workflow = self.create_graph()
        self.app = self.workflow.compile()

    def setup_components(self, pdf_path: str, tavily_api_key: str):
        # Load and process PDF
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250
        )
        self.texts = text_splitter.split_documents(documents)
        self.retriever = BM25Retriever.from_documents(self.texts, k=10)
        
        # Setup Tavily client
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def create_chains(self):
        # Create base prompt template with chat history
        base_prompt = """
        <START_PROMPT>
        Chat History:
        {chat_history}
        
        Current Query: {query}
        Context: {context}
        
        Anda adalah seorang agen yang akan membantu user terkait logistik dan pengiriman ekspor. Jawalah query user
        berdasarkan context yang diberikan.
        
        Jika relevan dengan pertanyaan anda dapat merekomendasikan disini, prioritaskan yang dekat dengan lokasi user maksimal 3:
        [daftar perusahaan logistik]
        <END_PROMPT>
        """
        
        self.prompt = PromptTemplate(
            input_variables=["query", "context", "chat_history"],
            template=base_prompt
        )

        # Create hallucination checker prompt
        hallucination_prompt = """
        <START_PROMPT>
        Query: {query}
        Answer: {answer}
        Context: {context}
        Anda adalah seorang agen yang akan membantu user terkait logistik dan pengiriman ekspor. 
        Evaluasi apakah jawaban yang diberikan adalah halusinasi dari query dan context.
        Hanya output antara true atau false satu kata tanpa hal lain
        <END_PROMPT>
        """
        
        self.hallucination_prompt = PromptTemplate(
            input_variables=["query", "answer", "context"],
            template=hallucination_prompt
        )

        # Initialize chains
        llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
        self.writer_chain = self.prompt | llm | StrOutputParser()
        self.grader_chain = self.hallucination_prompt | llm | StrOutputParser()

    def format_chat_history(self) -> str:
        """Format chat history for prompt inclusion"""
        if not self.chat_history:
            return "No previous conversation"
        
        formatted = []
        for i, (q, a) in enumerate(self.chat_history, 1):
            formatted.append(f"Q{i}: {q}\nA{i}: {a}")
        return "\n\n".join(formatted)

    def bm25_retriever_chain(self, query: str, k: int) -> str:
        """Retrieve relevant context using BM25"""
        retrieved_docs = self.retriever.invoke(query, top_k=k)
        content = ""
        for idx, doc in enumerate(retrieved_docs, 1):
            content += f"Chunk {idx}:\n{doc.page_content}\n{'-' * 80}\n"
        return content

    def tavily_retriever_chain(self, query: str) -> str:
        """Retrieve context from web using Tavily"""
        print('Mencari di Web:', query)
        context = self.tavily_client.search(
            query=query,
            search_depth='advanced',
            max_results=5,
            max_tokens=20000
        )
        return context

    def create_graph(self):
        """Create the workflow graph"""
        class GraphState(TypedDict):
            query: str
            context: str | None
            answer: str | None
            is_hallucination: bool | None
            chat_history: List[Tuple[str, str]]

        def retrieve_context(state: GraphState) -> GraphState:
            """Retrieve initial context using BM25."""
            print('[INFO] Mengambil informasi dari dokumen')
            context = self.bm25_retriever_chain(state['query'], k=5)
            return {"context": context}

        def get_initial_answer(state: GraphState) -> GraphState:
            """Get initial answer using the context."""
            chat_history = self.format_chat_history()
            answer = self.writer_chain.invoke({
                "query": state["query"],
                "context": state["context"],
                "chat_history": chat_history
            })
            return {"answer": answer}

        def check_hallucination(state: GraphState) -> GraphState:
            """Check if the answer is a hallucination."""
            result = self.grader_chain.invoke({
                "query": state["query"],
                "answer": state["answer"],
                "context": state["context"]
            })
            is_hallucination = result.lower() == "true"
            return {"is_hallucination": is_hallucination}

        def get_web_context(state: GraphState) -> GraphState:
            """Get additional context from web if needed."""
            print('[INFO] Mengambil informasi dari Web')
            web_results = self.tavily_retriever_chain(state["query"])
            context = f"Web Context:\n{web_results}"
            return {"context": context}

        def get_final_answer(state: GraphState) -> GraphState:
            """Get final answer and update chat history."""
            print('[INFO] Menulis jawaban akhir')
            chat_history = self.format_chat_history()
            answer = self.writer_chain.invoke({
                "query": state["query"],
                "context": state["context"],
                "chat_history": chat_history
            })
            
            # Update chat history
            self.chat_history.append((state["query"], answer))
            
            return {**state, "answer": answer}

        def router(state: GraphState) -> str:
            """Route based on hallucination check."""
            return "need_web" if state["is_hallucination"] else "final"

        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("get_initial_answer", get_initial_answer)
        workflow.add_node("check_hallucination", check_hallucination)
        workflow.add_node("get_web_context", get_web_context)
        workflow.add_node("get_final_answer", get_final_answer)

        # Add edges
        workflow.add_edge("retrieve_context", "get_initial_answer")
        workflow.add_edge("get_initial_answer", "check_hallucination")
        
        workflow.add_conditional_edges(
            "check_hallucination",
            router,
            {
                "need_web": "get_web_context",
                "final": "get_final_answer"
            }
        )
        
        workflow.add_edge("get_web_context", "get_final_answer")
        workflow.add_edge("get_final_answer", END)

        workflow.set_entry_point("retrieve_context")
        
        return workflow

    def chat(self, query: str) -> str:
        """Process a chat message and return the response"""
        inputs = {"query": query}
        final_state = None
        for output in self.app.stream(inputs):
    # print(output['get_final_answer'])
            for key, value in output.items():
                # pass
                # print('1 ', key)
                # print('2 ', value)

                if key == 'get_final_answer':
                    answer = (output[key]['answer'])
        # for output in self.app.stream(inputs):
        #     final_state = output
            
        return answer



agent = ExportChatAgent(
    pdf_path="C:\\Users\\ASUS\\Downloads\\hackathon-ai\\gov-ai\\export-umkm-agent\\important-docs\\modul-umkm-ekspor.pdf",
    tavily_api_key=os.environ.get('TAVILY_API_KEY')
)

# Chat with the agent
response1 = agent.chat("jelaskan npwp")
print(response1)

# response2 = agent.chat("tadi saya bertanya apa")
# print(response2)

# response3 = agent.chat("siapa alif bintang elfandra")
# print(response3)
