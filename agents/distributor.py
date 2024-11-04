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
        berdasarkan context yang diberikan. Jika relevan dengan percakapan anda dapat merekomendasikan lokasi ekspedisi disini, prioritaskan yang dekat dengan lokasi user maksimal 3. Pastikan hanya jika relevan
        "PT Samudera Indonesia Tbk","Jl. Letjen S. Parman No.Kav 35, Jakarta Barat 11480"
        "PT Lautan Luas Tbk","Graha Indramas, Jl. Aipda Ks Tubun No.77, Jakarta Barat 11410"
        "PT Berlian Laju Tanker Tbk","Wisma BSG 5th Floor, Jl. Abdul Muis No.40, Jakarta Pusat 10160"
        "PT Agility International","Wisma Soewarna 1st Floor, Soewarna Business Park, Soekarno-Hatta International Airport, Tangerang 15126"
        "PT Schenker Petrolog Utama (DB Schenker)","Deutsche Bank Building, Tower 2, 13th Floor, Jl. Imam Bonjol No.80, Jakarta Pusat 10310"
        "PT Yusen Logistics Indonesia","Midplaza 2 Building, 15th Floor, Jl. Jenderal Sudirman Kav.10-11, Jakarta Pusat 10220"
        "PT Damco Indonesia","Gedung Bumi Mandiri Tower II, 12th Floor, Jl. Panglima Sudirman No.66-68, Surabaya 60271"
        "PT Kintetsu World Express Indonesia","Soewarna Business Park Block A Lot 5, Soekarno-Hatta International Airport, Tangerang 15126"
        "PT DHL Global Forwarding Indonesia","Gedung TIFA Arum Realty Lt.2, Jl. Kuningan Barat I No.26, Jakarta Selatan 12710"
        "PT Kuehne + Nagel Indonesia","Graha Inti Fauzi, 5th Floor, Jl. Buncit Raya No.22, Jakarta Selatan 12510"
        "PT CEVA Logistics Indonesia","Wisma 46 Kota BNI, 38th Floor, Jl. Jenderal Sudirman Kav.1, Jakarta Pusat 10220"
        "PT Expeditors Indonesia","Wisma GKBI 9th Floor, Suite 903, Jl. Jenderal Sudirman No.28, Jakarta Pusat 10210"
        "PT FedEx Express Indonesia","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT TNT Skypak International Express","Jl. Jalur Sutera Timur Kav.7A No.15, Alam Sutera, Tangerang 15143"
        "PT Panalpina Nusajaya Transport","Jl. Raya Protokol Halim Perdanakusuma, Jakarta Timur 13610"
        "PT UPS Cardig International","Gedung UPS, Jl. Raya Protokol Halim Perdanakusuma, Jakarta Timur 13610"
        "PT APL Logistics","Menara Rajawali 15th Floor, Jl. DR. Ide Anak Agung Gde Agung Lot #5.1, Kawasan Mega Kuningan, Jakarta Selatan 12950"
        "PT Bolloré Logistics Indonesia","Soewarna Business Park Block A, Lot 1-2, Soekarno-Hatta International Airport, Tangerang 15126"
        "PT Nippon Express Indonesia","Wisma KEIAI 15th Floor, Jl. Jenderal Sudirman Kav.3, Jakarta Pusat 10220"
        "PT Wahana Prestasi Logistik","Jl. Rempoa Raya No.88, Ciputat Timur, Tangerang Selatan 15412"
        "PT Toll Indonesia","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Dimerco Express Indonesia","Soewarna Business Park Block A, Lot 8, Soekarno-Hatta International Airport, Tangerang 15126"
        "PT Hellmann Worldwide Logistics Indonesia","Menara Jamsostek, North Tower, 11th Floor, Jl. Jenderal Gatot Subroto No.38, Jakarta Selatan 12710"
        "PT Hitachi Transport System Indonesia","Jl. Raya Narogong Km.12,5 Pangkalan 6, Bekasi 17310"
        "PT Sankyu Indonesia International","Jl. Harapan Raya Lot KK-7, Kawasan Industri KIIC, Karawang 41361"
        "PT Siba Surya","Jl. Raya Kaligawe Km.5, Genuk, Semarang 50115"
        "PT Djakarta Lloyd (Persero)","Gedung Djakarta Lloyd, Jl. Ir. H. Juanda No.10, Jakarta Pusat 10120"
        "PT Kamadjaja Logistics","Jl. Greges Jaya I No.1, Asemrowo, Surabaya 60183"
        "PT Megacom Translogistics","Jl. Raya Bogor Km.22/5, Ciracas, Jakarta Timur 13740"
        "PT H & M Transportation","Jl. Rawa Kepiting No.6, Kawasan Industri Pulogadung, Jakarta Timur 13920"
        "PT Siba Cargo","Jl. Arteri Yos Sudarso No.1, Semarang 50174"
        "PT Samudera Lintas Lautan","Jl. Enggano No.38, Tanjung Priok, Jakarta Utara 14310"
        "PT Cipta Krida Bahari (CKB Logistics)","CKB Logistics Center, Jl. Raya Cakung Cilincing Kav.3A, Jakarta Utara 14130"
        "PT Pos Indonesia (Persero)","Jl. Cilaki No.73, Bandung 40115"
        "PT Duta Logistics","Jl. Raya Bandara Juanda Km.3, Sedati, Sidoarjo 61253"
        "PT Asiana Express Indonesia","Jl. Puri Kencana No.1, Kembangan, Jakarta Barat 11610"
        "PT Evergreen Shipping Agency Indonesia","Gedung Graha Kirana Lt. 10, Jl. Yos Sudarso Kav.88, Jakarta Utara 14350"
        "PT Pelayaran Tempuran Emas Tbk (Temas Line)","Gedung Temas, Jl. Yos Sudarso Kav.33, Jakarta Utara 14350"
        "PT Tiki Jalur Nugraha Ekakurir (JNE)","Jl. Tomang Raya No.11, Jakarta Barat 11440"
        "PT Pandu Siwi Sentosa (Pandu Logistics)","Jl. Raya Perancis No.67, Dadap, Tangerang 15211"
        "PT Cardig Logistics Indonesia","Soewarna Business Park Block H Lot 1-2, Soekarno-Hatta International Airport, Tangerang 15126"
        "PT Buana Centra Swakarsa","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Birotika Semesta (DHL Express)","Ruko Grand Tomang Blok A1-A2, Jl. Tomang Raya, Jakarta Barat 11440"
        "PT Indah Logistik","Jl. Letjen Suprapto No.7, Cempaka Putih, Jakarta Pusat 10520"
        "PT Maersk Indonesia","World Trade Center II, 10th Floor, Jl. Jenderal Sudirman Kav.29-31, Jakarta Selatan 12920"
        "PT MSC Indonesia","Gedung Artha Graha 9th Floor, Jl. Jenderal Sudirman Kav.52-53, Jakarta Selatan 12190"
        "PT NYK Line Indonesia","Midplaza 1, 16th Floor, Jl. Jenderal Sudirman Kav.10-11, Jakarta Pusat 10220"
        "PT COSCO Shipping Lines Indonesia","Menara Batavia, 21st Floor, Jl. KH Mas Mansyur Kav.126, Jakarta Pusat 10220"
        "PT CMA CGM Indonesia","Sentral Senayan II, 17th Floor, Jl. Asia Afrika No.8, Jakarta Pusat 10270"
        "PT Yang Ming Shipping Indonesia","Menara Kadin Indonesia, 25th Floor, Jl. HR Rasuna Said Blok X-5 Kav.2-3, Jakarta Selatan 12950"
        "PT OOCL Indonesia","Wisma 46 Kota BNI, 27th Floor, Jl. Jenderal Sudirman Kav.1, Jakarta Pusat 10220"
        "PT Hapag-Lloyd Indonesia","Menara Anugrah, 26th Floor, Jl. Mega Kuningan Lot 8.6-8.7, Jakarta Selatan 12950"
        "PT Eculine Indonesia","Jl. Gunung Sahari Raya No.2, Jakarta Pusat 10720"
        "PT Damai Indah Utama","Jl. Raya Pluit Selatan No.103, Jakarta Utara 14450"
        "PT Air & Sea Transport","Jl. Raya Kuta No.88, Kuta, Bali 80361"
        "PT Pelabuhan Indonesia II (Persero)","Jl. Pasoso No.1, Tanjung Priok, Jakarta Utara 14310"
        "PT Salam Pacific Indonesia Lines (SPIL)","Jl. Perak Timur No.620, Surabaya 60164"
        "PT Tanto Intim Line","Jl. Enggano No.31, Tanjung Priok, Jakarta Utara 14310"
        "PT Samudera Sarana Logistik","Jl. Kali Besar Barat No.27, Jakarta Barat 11230"
        "PT Aero Express Indonesia","Jl. Raya Bandara Halim Perdanakusuma, Jakarta Timur 13610"
        "PT Citra Van Titipan Kilat (TIKI)","Jl. Raden Saleh Raya No.2, Cikini, Jakarta Pusat 10330"
        "PT Puninar Logistics","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Buana Raya Cargo","Jl. Raya Kebayoran Lama No.121, Jakarta Selatan 12210"
        "PT Royalindo Expoduta","Jl. Pintu Air Raya No.28, Jakarta Pusat 10710"
        "PT Linc Group","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Siba Logistics","Jl. Raya Kaligawe Km.5, Genuk, Semarang 50115"
        "PT Pahala Kencana Transindo","Jl. Raya Pasar Minggu No.7, Jakarta Selatan 12740"
        "PT Ritra Cargo Indonesia","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Sarana Karya Utama Transindo","Jl. Raya Narogong Km.12,5, Bekasi 17310"
        "PT Sriwijaya Air Cargo","Jl. Marsekal Suryadarma No.5, Tangerang 15129"
        "PT Pos Logistik Indonesia","Jl. Lapangan Banteng Utara No.1, Jakarta Pusat 10710"
        "PT Indonesia AirAsia Extra","Jl. Prof. DR. Soepomo SH No.45, Tebet, Jakarta Selatan 12810"
        "PT J&T Express","Jl. Pluit Selatan Raya No.2, Penjaringan, Jakarta Utara 14450"
        "PT Lion Parcel","Lion Air Tower, Jl. Gajah Mada No.7, Jakarta Pusat 10130"
        "PT Pahala Express","Jl. Raya Pasar Minggu No.7, Jakarta Selatan 12740"
        "PT Trans Continent","Komplek Ruko CBD Polonia Blok GG No.8, Medan 20157"
        "PT Anugerah Lintas Samudera","Jl. Kalimas Baru No.189, Surabaya 60165"
        "PT Daya Nusa Trans","Jl. Raya Imam Bonjol No.28, Karawaci, Tangerang 15115"
        "PT Multisarana Bahtera","Jl. Letjen Suprapto Kav.1, Jakarta Pusat 10640"
        "PT Global Putra International Group","Jl. Pulo Kambing II No.33, Kawasan Industri Pulogadung, Jakarta Timur 13930"
        "PT First Logistics","Jl. Raya Cilandak KKO No.101, Jakarta Selatan 12560"
        "PT Batamfast Indonesia","Harbour Bay Ferry Terminal, Jl. Duyung, Sei Jodoh, Batam 29453"
        "PT Samudera Perdana Selaras","Jl. Raya Gubeng No.19, Surabaya 60281"
        "PT Siba Mandiri","Jl. Raya Kaligawe Km.5, Genuk, Semarang 50115"
        "PT Hiba Utama","Jl. Raya Bekasi Km.22, Cakung, Jakarta Timur 13910"
        "PT Serasi Autoraya (SERA)","Jl. Mitra Sunter Boulevard Blok D3 No.1, Sunter, Jakarta Utara 14330"
        "PT MNC Logistic","MNC Tower, Jl. Kebon Sirih No.17-19, Jakarta Pusat 10340"
        "PT Wira Logistics","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Sentral Cargo","Jl. Raya Pos Pengumben No.72, Kebon Jeruk, Jakarta Barat 11560"
        "PT Eka Sari Lorena Express","Jl. KH Hasyim Ashari No.125, Jakarta Pusat 10150"
        "PT GAC Samudera Logistics","MM2100 Industrial Town, Block A3-1, Cikarang Barat, Bekasi 17520"
        "PT Pacific International Lines (PIL) Indonesia","Menara Standard Chartered, 30th Floor, Jl. Prof. DR. Satrio No.164, Jakarta Selatan 12930"
        "PT Samudera Naga Global","Jl. Pluit Selatan Raya No.103, Jakarta Utara 14450"
        "PT Mitra Sentosa Logistik","Jl. Raya Cakung Cilincing Km.1,5, Jakarta Utara 14130"
        "PT Berlian Jasa Terminal Indonesia","Jl. Perak Barat No.379, Surabaya 60165"
        "PT Andalan Express Indonesia","Jl. Raya Pasar Minggu No.7, Jakarta Selatan 12740"
        "PT Nusantara Card Semesta (NCS)","Jl. Kebon Bawang IX No.5, Tanjung Priok, Jakarta Utara 14320"
        "PT Berlian Dumai Logistics","Jl. Soekarno-Hatta No.1, Dumai 28826"
        "PT Kargo Indonesia","Jl. Kemang Timur No.69, Jakarta Selatan 12730"
        "PT Adi Sarana Armada Tbk (ASSA)","Jl. Inspeksi Kalimalang No.Kav.1, Jakarta Timur 13450"
        

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
