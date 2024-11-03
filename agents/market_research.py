from typing import TypedDict, List, Optional
from langgraph.graph import START, END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient
from IPython.display import display, Markdown
import ast

class GraphState(TypedDict):
    product_description: str
    outline: Optional[str]
    context: str
    iteration_count: int
    queries: Optional[List[str]]
    is_continue: Optional[str]
    searched_query: Optional[List[str]]
    result: Optional[str]

def create_planner_chain():
    prompt_template = """
    <START_PROMPT>
    Anda adalah agen AI yang bertugas membuat struktur laporan dan SWOT (Strengths, Weaknesses, Opportunities, Threats) berdasarkan deskripsi produk yang diberikan untuk tujuan ekspor.
    Tujuan Anda adalah memberikan outline lengkap untuk laporan akhir dan analisis SWOT, termasuk bagian-bagian utama dan tiga informasi spesifik yang harus dikumpulkan untuk setiap bagian.
    Laporan ini ditujukan untuk usaha Indonesia yang ingin ekspor ke ASEAN, jadi fokuskan untuk informasi ekspor ke ASEAN. Di bagian atas, sebutkan bahwa ini untuk Ekspor ke ASEAN.
    Pastikan tulis dalam format markdown. Tugas Anda hanya memberi outline yakni informasi yang perlu dicari tau, anda tidak bertugas menjawab.
    
    ## SWOT Analysis
    ### Strengths
    - Informasi 1
    - Informasi 2
    - dst

    ### Weaknesses
    - Informasi 1
    - Informasi 2
    - dst

    ### Opportunities
    - Informasi 1
    - Informasi 2
    - dst

    ### Threats
    - Informasi 1
    - Informasi 2
    - dst

    ## STP (Segmenting, Targeting, Positioning) (Hanya 1 Bagian)
    - Informasi 1
    - Informasi 2
    - dst

    Deskripsi Produk: {product_description}
    Jawaban: <START_RESPONSE>
    """
    prompt = PromptTemplate(input_variables=["product_description"], template=prompt_template)
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    return prompt | llm | StrOutputParser()

def create_web_planner_chain(num_query: int = 4):
    prompt_template = """
    Deskripsi Produk: {product_description}
    Outline: {outline}
    Informasi yang Sudah Diperoleh dari Iterasi Sebelumnya: {context}
    Nomor Iterasi Saat Ini: {iteration_count}
    Query yang sudah dicari: {searched_query}

    Instruksi:
    1. Hanya buat query untuk bagian outline yang belum terisi atau belum lengkap berdasarkan hasil dari iterasi sebelumnya.
    2. Buat Query dengan bahasa Indonesia dan juga Inggris, Jadi ada yang Bahasa Indonesia dan Inggris.
    3. Buat query dengan gaya dan kedalaman yang berbeda, mencakup aspek umum dan spesifik sesuai kebutuhan outline.
    4. Jangan ambil query yang sama dengan yang sudah di cari.
    5. Semua bagian informasi maksimal 2 dan jangan terlalu panjang. Pastikan setiap query singkat dan padat.

    Output dalam format list contohnya:
    ["query_1", "query_2", "query_3"]
    Saya ulangi, dalam format list ini sangat penting untuk karir saya hanya output list saja.
    """

    prompt_template += f"\nBerdasarkan deskripsi produk dan outline, serta informasi yang telah dikumpulkan di iterasi sebelumnya, buatlah {num_query} query untuk Tavily API yang berfokus pada informasi yang masih kurang."

    prompt = PromptTemplate(
        input_variables=[ "outline", "context", "iteration_count", "searched_query"],
        template=prompt_template
    )
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    return prompt | llm | StrOutputParser()

def create_grader_chain():
    prompt_template = """
    <START_PROMPT>
    Deskripsi Produk: {product_description}
    Outline: {outline}
    Informasi yang Terkumpul: {context}

    Tugas Anda adalah menilai apakah informasi yang terkumpul dalam context sudah memenuhi semua poin yang ada di outline. 
    Periksa tiap bagian dari outline dan pastikan apakah informasi yang dibutuhkan sudah lengkap atau masih ada yang kurang.

    Jika semua poin di outline sudah terisi dengan informasi yang sesuai, beri hasil "false". 
    Jika masih ada poin yang kurang atau belum lengkap, beri hasil "true".

    Jawaban dalam format String hanya bisa:
    true atau false
    PERHATIKAN hanya boleh true atau false saja pastikan huruf kecil
    <END_PROMPT>
    """
    prompt = PromptTemplate(input_variables=["product_description", "outline", "context"], template=prompt_template)
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    return prompt | llm | StrOutputParser()

def create_writer_chain():
    prompt_template = """
    <START_PROMPT>
    Deskripsi Produk: {product_description}
    Outline: {outline}
    Informasi Terkumpul (Context): {context}

    Tugas Anda adalah menuliskan laporan akhir berdasarkan outline yang sudah diberikan. 
    Ikuti instruksi berikut. Informasi ini akan digunakan untuk membantu Ekspor ke Luar Negeri:

    1. Tulis laporan dalam format paragraf untuk setiap bagian outline, tanpa menggunakan bullet point.
    2. Saat mengambil informasi dari Context, sertakan referensi atau sumbernya, tulis dalam hyperlink.
    3. Anda dapat menambahkan analisis tambahan berdasarkan wawasan Anda sendiri, asalkan tetap relevan dan selaras dengan Context yang diberikan.
    4. Pastikan setiap bagian dari laporan sesuai dengan struktur outline yang diberikan.
    5. Anda harus menulis dalam markdown
    6. Untuk bagian SWOT buat menjadi 4 bagian, untuk bagian STP menjadi 1 pagaraf saja
    <END_PROMPT>
    """
    prompt = PromptTemplate(input_variables=["product_description", "outline", "context"], template=prompt_template)
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    return prompt | llm | StrOutputParser()

def create_workflow(tavily_client, num_query: int = 4, num_tavily: int = 5, max_iteration: int = 3):
    planner_chain = create_planner_chain()
    tavily_planner_chain = create_web_planner_chain(num_query)
    grader_chain = create_grader_chain()
    writer_chain = create_writer_chain()

    def planner(state):
        print('[INFO] 📝 Memulai proses perencanaan (Planner)')
        product_description = state['product_description']
        outline = planner_chain.invoke({'product_description': product_description})
        print('[INFO] ✅ Planner selesai menghasilkan outline')
        return {
            'product_description': product_description,
            'outline': outline,
            'context': '',
            'iteration_count': 1,
            'is_continue': 'true',
            'searched_query': []
        }

    def web_planner(state):
        print('[INFO] 🔍 Memulai perencanaan pencarian informasi (Web Planner)')
        queries = ast.literal_eval(tavily_planner_chain.invoke(state))
        print('[INFO] 📋 Web Planner menghasilkan query pencarian')
        return {
            'queries': queries,
            'searched_query': state['searched_query'] + queries,
            'iteration_count': state['iteration_count'] + 1
        }

    def web_retriever(state):
        print('[INFO] 🌐 Memulai proses pengambilan data dari web (Web Retriever)')
        context = []
        for query in state['queries']:
            print('Mencari di Web:', query)
            results = tavily_client.search(query=query, search_depth='advanced', max_results=num_tavily, max_tokens=20000)
            for result in results['results']:
                formatted_result = f"{result['url']} - {result['content']}"
                context.append(formatted_result)
        
        new_context = state['context'] + "\n" + "\n".join(context)
        print('[INFO] 🔗 Web Retriever berhasil memperbarui konteks')
        return {'context': new_context}

    def grader(state):
        print('[INFO] 📊 Memulai proses penilaian (Grader)')
        if state['iteration_count'] > max_iteration:
            is_continue = 'false'
            print('[INFO] ⏹️ Iterasi maksimum tercapai. Menghentikan proses iterasi.')
        else:
            is_continue = grader_chain.invoke(state)
            print(f'[INFO] 🧮 Grader menentukan kelanjutan: {is_continue}')
        return {'is_continue': is_continue}

    def writer(state):
        print('[INFO] ✍️ Memulai proses penulisan laporan akhir (Writer)')
        result = writer_chain.invoke(state)
        print('[INFO] 📄 Writer menghasilkan laporan akhir')
        display(Markdown(result))
        return {'result': result}

    def decide_to_continue(state):
        return 'web_planner' if state['is_continue'] == 'true' else 'writer'

    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("web_planner", web_planner)
    workflow.add_node("web_retriever", web_retriever)
    workflow.add_node("grader", grader)
    workflow.add_node("writer", writer)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "web_planner")
    workflow.add_edge("web_planner", "web_retriever")
    workflow.add_edge("web_retriever", "grader")
    workflow.add_conditional_edges("grader", decide_to_continue, {
        "web_planner": "web_planner",
        "writer": "writer",
    })
    workflow.add_edge("writer", END)
    
    return workflow.compile()

def market_research(product_description: str, num_query: int = 1, num_tavily: int = 5, max_iteration: int = 1):
    """
    Performs market research based on the given query and parameters.
    
    Args:
        query (str): The product or topic to research
        num_query (int, optional): Number of search queries to generate per iteration. Defaults to 4.
        num_tavily (int, optional): Number of results to fetch per query. Defaults to 5.
        max_iteration (int, optional): Maximum number of research iterations. Defaults to 3.
    """
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key="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")
    
    # Create and run workflow
    app = create_workflow(tavily_client, num_query, num_tavily, max_iteration)
    inputs = {"product_description": product_description}
    
    # Execute workflow and process results
    for output in app.stream(inputs):
        for key, value in output.items():
            pass  # Results are displayed through the workflow's print statements
    

market_research('rempah rempah khas sumatera',1,5,1)