from typing import TypedDict, List, Optional
from langgraph.graph import START, END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from IPython.display import display, Markdown
from tavily import TavilyClient
import ast
import openai
# Type Definitions
class GraphState(TypedDict):
    product_description: str                        
    outline: Optional[str]                          
    context: str                                   
    iteration_count: int                           
    queries: Optional[List[str]]                    
    is_continue: Optional[str]   
    searched_query: Optional[List[str]]                  
    result: Optional[str]                          

# Chain Creation Functions
def create_planner_chain():
    template = """
    Anda adalah agen AI yang bertugas membantu memberikan rekomendasi dan tips pengembangan produk (Product-R&D) untuk tujuan ekspor berdasarkan deskripsi produk yang diberikan. 
    Tujuan Anda adalah memberikan outline lengkap yang mencakup tips inovasi produk, efisiensi produksi, dan rekomendasi peningkatan kualitas produk agar sesuai dengan standar ekspor ASEAN. 
    Laporan ini ditujukan untuk usaha Indonesia yang ingin mengekspor ke ASEAN, jadi fokuskan rekomendasi Anda untuk pasar ASEAN.
    Pastikan tulis dalam format markdown. Tugas Anda hanya memberi outline yakni informasi yang perlu dicari tau, anda tidak bertugas menjawab, saya ulangi, anda tidak bertugas menjawab.
    Contoh struktur laporan seperti ini:

    ## Product-R&D Recommendations
    ### Innovation Tips 
    - Informasi 1
    - Informasi 2

    ### Production Efficiency
    - Informasi 1
    - Informasi 2

    ### Quality Enhancement
    - Informasi 1
    - Informasi 2

    Ikuti ini, Semua bagian informasi maksimal 2 dan sangat singkat saja. Tidak perlu menuliskan informasi secara eksplisit
    Deskripsi Produk: {product_description}
    Jawaban: 
    """
    prompt = PromptTemplate(input_variables=["product_description"], template=template)
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    return prompt | llm | StrOutputParser()

def create_tavily_planner_chain(num_query: int = 4):
    template = template = """
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

    """ + f"Berdasarkan deskripsi produk dan outline, serta informasi yang telah dikumpulkan di iterasi sebelumnya, buatlah {num_query} query untuk Tavily API yang berfokus pada informasi yang masih kurang."
    prompt = PromptTemplate(
        input_variables=["product_description", "outline", "context", "iteration_count", "searched_query"],
        template=template
    )
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    return prompt | llm | StrOutputParser()

def create_tavily_retriever(num_results: int):
    tavily_client = TavilyClient(api_key="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")
    
    def tavily_retriever_chain(queries):
        context = []
        for query in queries:
            print('Mencari di Web:', query)
            cur_context = tavily_client.search(
                query=query, 
                search_depth='advanced', 
                max_results=num_results,
                max_tokens=20000
            )
            
            for result in cur_context['results']:
                url = result.get('url', '')
                content = result.get('content', '')
                formatted_result = f"{url} - {content}"
                # print(formatted_result)
                context.append(formatted_result)
                
        return "\n".join(context)
    
    return tavily_retriever_chain

def create_grader_chain():
    template = """
    Deskripsi Produk: {product_description}
    Outline: {outline}
    Informasi yang Terkumpul: {context}

    Tugas Anda adalah menilai apakah informasi yang terkumpul dalam context sudah memenuhi semua poin yang ada di outline. Periksa tiap bagian dari outline dan pastikan apakah informasi yang dibutuhkan sudah lengkap atau masih ada yang kurang.

    Jika semua poin di outline sudah terisi dengan informasi yang sesuai, beri hasil "false". Jika masih ada poin yang kurang atau belum lengkap, beri hasil "true".

    Jawaban dalam format String hanya bisa:
    true atau false
    PERHATIKAN hanya boleh true atau false saja pastikan huruf kecil
    """
    prompt = PromptTemplate(
        input_variables=["product_description", "outline", "context"],
        template=template
    )
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    return prompt | llm | StrOutputParser()

def create_writer_chain(primary_model='gpt-4o', fallback_model='gpt-4o-mini'):
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
    6. Untuk bagian SWOT buat menjadi 4 bagian, untuk bagian STP menjadi 1 paragraf saja
    <END_PROMPT>
    """
    prompt = PromptTemplate(input_variables=["product_description", "outline", "context"], template=prompt_template)
    llm = ChatOpenAI(model_name=primary_model, temperature=0)
    return prompt | llm | StrOutputParser()

def create_fallback_writer_chain():
    return create_writer_chain(primary_model='gpt-4o')

def create_workflow(planner_chain, tavily_planner_chain, tavily_retriever_chain, grader_chain, writer_chain, max_iteration: int):
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
        queries = ast.literal_eval(tavily_planner_chain.invoke({
            'product_description': state['product_description'], 
            'outline': state['outline'], 
            'context': state['context'], 
            'iteration_count': state['iteration_count'],
            'searched_query': state['searched_query'],
            
        }))
        print('[INFO] 📋 Web Planner menghasilkan query pencarian')
        return {
            'queries': queries,
            'searched_query': state['searched_query'] + queries,
            'iteration_count': state['iteration_count'] + 1
        }

    def web_retriever(state):
        print('[INFO] 🌐 Memulai proses pengambilan data dari web (Web Retriever)')
        new_context = tavily_retriever_chain(state['queries'])
        context = state['context'] + "\n" + new_context
        print('[INFO] 🔗 Web Retriever berhasil memperbarui konteks')
        return {'context': context}

    def grader(state):
        print('[INFO] 📊 Memulai proses penilaian (Grader)')
        if state['iteration_count'] > max_iteration:
            is_continue = 'false'
            print('[INFO] ⏹️ Iterasi maksimum tercapai. Menghentikan proses iterasi.')
        else:
            is_continue = grader_chain.invoke({
                'product_description': state['product_description'], 
                'outline': state['outline'], 
                'context': state['context']
            })
            print(f'[INFO] 🧮 Grader menentukan kelanjutan: {is_continue}')
        return {'is_continue': is_continue}

    def writer(state):
        print('[INFO] ✍️ Memulai proses penulisan laporan akhir (Writer)')
        try:
            result = writer_chain.invoke(state)
            print('[INFO] 📄 Writer menghasilkan laporan akhir')
            print(result)
            return {'result': result}
        except openai.RateLimitError:
            print('[WARNING] 🚨 Rate limit exceeded on primary model. Switching to fallback model (gpt-4o).')
            fallback_writer_chain = create_fallback_writer_chain()
        try:
            result = fallback_writer_chain.invoke(state)
            print('[INFO] 📄 Fallback writer menghasilkan laporan akhir')
            print(result)
            return {'result': result}
        except openai.RateLimitError:
            print('[ERROR] ❌ Rate limit exceeded on both primary and fallback models. Please try again later.')
            return {'result': None}

    def decide_to_continue(state):
        is_continue = state['is_continue']
        if is_continue == 'true':
            print('[INFO] 🔄 Keputusan: Melanjutkan dengan Web Planner')
            return 'web_planner'
        else:
            print('[INFO] 🏁 Keputusan: Melanjutkan ke tahap Writer')
            return 'writer'

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
    workflow.add_conditional_edges(
        "grader",
        decide_to_continue,
        {
            "web_planner": "web_planner",
            "writer": "writer",
        },
    )
    workflow.add_edge("writer", END)
    
    return workflow.compile()

def product_rnd(product_description: str, num_query: int = 4, num_tavily: int = 5, max_iteration: int = 4):
    """
    Execute product R&D workflow for export recommendations
    
    Args:
        product_description (str): Description of the product
        num_query (int): Number of queries to generate per iteration
        num_tavily (int): Number of results to fetch from Tavily API
        max_iteration (int): Maximum number of iterations
        
    Returns:
        dict: Final state of the workflow including results
    """
    # Create chains
    planner_chain = create_planner_chain()
    tavily_planner_chain = create_tavily_planner_chain(num_query)
    tavily_retriever_chain = create_tavily_retriever(num_tavily)
    grader_chain = create_grader_chain()
    writer_chain = create_writer_chain()
    
    # Create and run workflow
    app = create_workflow(
        planner_chain,
        tavily_planner_chain,
        tavily_retriever_chain,
        grader_chain,
        writer_chain,
        max_iteration
    )
    inputs = {"product_description": product_description}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pass  

product_rnd('Minyak atsiri alami yang diekstraksi dari tumbuhan pilihan untuk menghasilkan aroma yang murni dan berkualitas tinggi.',4,5,4)