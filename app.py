import streamlit as st
import re
import ast
import json
from typing import TypedDict, List, Optional
from pprint import pprint

# Import untuk LangChain dan komponen lainnya
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph

# Import TavilyClient
from tavily import TavilyClient

# Judul aplikasi
st.title("UMKM EXPORT AGENT: Laporan STP, Pengembangan Produk, dan Ekspansi Pasar bagi UMKM")

# Meminta deskripsi produk dari pengguna
product_description = st.text_area("Masukkan deskripsi produk Anda:")

# Tombol untuk memulai proses
if st.button("Generate Report"):
    # Validasi Deskripsi Produk
    if not product_description.strip():
        st.error("Harap masukkan deskripsi produk Anda.")
    else:
        # Mengambil API key dari st.secrets
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        tavily_api_key = st.secrets["TAVILY_API_KEY"]

        # Inisialisasi TavilyClient dengan API key dari secrets
        tavily_client = TavilyClient(api_key=tavily_api_key)

        # Placeholder untuk log dan output
        log_placeholder = st.empty()
        output_placeholder = st.empty()

        # Inisialisasi daftar log
        logs = []

        # Definisi fungsi-fungsi
        def planner(state):
            logs.append('[INFO] 📝 Memulai proses perencanaan (Planner)')
            product_description = state['product_description']
            outline = planner_chain.invoke({'product_description': product_description})
            logs.append('[INFO] ✅ Planner selesai menghasilkan outline')
            log_placeholder.text("\n".join(logs))

            return {
                'product_description': product_description,
                'outline': outline,
                'context': '',
                'iteration_count': 1,
                'is_continue': 'true',
                'searched_query': []
            }

        def web_planner(state):
            logs.append('[INFO] 🔍 Memulai perencanaan pencarian informasi (Web Planner)')
            product_description = state['product_description']
            outline = state['outline']
            context = state['context']
            searched_query = state['searched_query']
            iteration_count = state['iteration_count']

            queries_str = tavily_planner_chain.invoke({
                'product_description': product_description,
                'outline': outline,
                'context': context,
                'iteration_count': iteration_count,
                'searched_query': searched_query
            })
            queries = ast.literal_eval(queries_str)
            logs.append('[INFO] 📋 Web Planner menghasilkan query pencarian')
            log_placeholder.text("\n".join(logs))

            return {
                'queries': queries,
                'searched_query': searched_query + queries,
                'iteration_count': iteration_count + 1
            }

        def tavily_retriever_chain(queries):
            context = []
            for query in queries:
                logs.append(f'[INFO] 🔍 Mencari di Web: {query}')
                cur_context = tavily_client.search(query=query, search_depth='advanced', max_results=3, max_tokens=10000)

                for result in cur_context['results']:
                    url = result.get('url', '')
                    content = result.get('content', '')
                    formatted_result = f"{url} - {content}"
                    logs.append(formatted_result)
                    context.append(formatted_result)

                log_placeholder.text("\n".join(logs))

            return "\n".join(context)

        def web_retriever(state):
            logs.append('[INFO] 🌐 Memulai proses pengambilan data dari web (Web Retriever)')
            queries = state['queries']
            context = state['context']

            new_context = tavily_retriever_chain(queries)
            context += "\n" + new_context
            logs.append('[INFO] 🔗 Web Retriever berhasil memperbarui konteks')
            log_placeholder.text("\n".join(logs))

            return {'context': context}

        def grader(state):
            logs.append('[INFO] 📊 Memulai proses penilaian (Grader)')
            product_description = state['product_description']
            outline = state['outline']
            context = state['context']
            iteration_count = state['iteration_count']

            if iteration_count > 4:
                is_continue = 'false'
                logs.append('[INFO] ⏹️ Iterasi maksimum tercapai. Menghentikan proses iterasi.')
            else:
                is_continue = grader_chain.invoke({
                    'product_description': product_description,
                    'outline': outline,
                    'context': context
                })
                logs.append(f'[INFO] 🧮 Grader menentukan kelanjutan: {is_continue}')

            log_placeholder.text("\n".join(logs))
            return {'is_continue': is_continue}

        def writer(state):
            logs.append('[INFO] ✍️ Memulai proses penulisan laporan akhir (Writer)')
            product_description = state['product_description']
            outline = state['outline']
            context = state['context']

            result = writer_chain.invoke({
                'product_description': product_description,
                'outline': outline,
                'context': context
            })
            logs.append('[INFO] 📄 Writer menghasilkan laporan akhir')
            log_placeholder.text("\n".join(logs))
            st.markdown(result)
            return {'result': result}

        def decide_to_continue(state):
            is_continue = state['is_continue']
            if is_continue == 'true':
                logs.append('[INFO] 🔄 Keputusan: Melanjutkan dengan Web Planner')
                log_placeholder.text("\n".join(logs))
                return 'web_planner'
            else:
                logs.append('[INFO] 🏁 Keputusan: Melanjutkan ke tahap Writer')
                log_placeholder.text("\n".join(logs))
                return 'writer'

        # Definisi kelas GraphState
        class GraphState(TypedDict):
            product_description: str
            outline: Optional[str]
            context: str
            iteration_count: int
            queries: Optional[List[str]]
            is_continue: Optional[str]
            searched_query: Optional[List[str]]
            result: Optional[str]

        # Inisialisasi LLM dan rantai-rantai
        llm = ChatOpenAI(model_name='gpt-4o', temperature=0, openai_api_key=openai_api_key)

        # Planner Chain
        planner_prompt_template = """
        <START_PROMPT>
        Anda adalah agen AI yang bertugas membuat struktur laporan STP (Segmenting, Targeting, Positioning) berdasarkan deskripsi produk yang diberikan untuk tujuan ekspor.
        Tujuan Anda adalah memberikan outline lengkap untuk laporan akhir STP, termasuk bagian-bagian utama dan tiga informasi spesifik yang harus dikumpulkan untuk setiap bagian.
        Outline ini akan menjadi acuan bagi Tavily Planner dalam mengumpulkan data yang dibutuhkan untuk setiap bagian. Informasi ini akan diberikan kepada usaha Indonesia yang ingin ekspor ke ASEAN, jadi fokuskan untuk informasi ekspor ke ASEAN. Di bagian atas sebutkan bahwa ini untuk Ekspor ke ASEAN.
        Pastikan tulis dalam format markdown.
        Contoh struktur laporan STP seperti ini:

        1. Segmenting: 
           - Informasi 1
           - Informasi 2
           - dst

        2. Targeting:
           - Informasi 1
           - Informasi 2
           - dst

        3. Positioning:
           - Informasi 1
           - Informasi 2
           - dst

        4. Saran Pengembangan
           - Informasi 1
           - Informasi 2

        5. Saran Negara Potensial
           - Informasi 1
           - Informasi 2

        Deskripsi Produk: {product_description}
        Jawaban: <START_RESPONSE>
        """

        planner_prompt = PromptTemplate(
            input_variables=["product_description"],
            template=planner_prompt_template
        )
        planner_chain = planner_prompt | llm | StrOutputParser()

        # Tavily Planner Chain
        tavily_planner_prompt_template = """
        <START_PROMPT>
        Deskripsi Produk: {product_description}
        Outline STP: {outline}
        Informasi yang Sudah Diperoleh dari Iterasi Sebelumnya: {context}
        Nomor Iterasi Saat Ini: {iteration_count}
        Query yang sudah dicari: {searched_query}

        Berdasarkan deskripsi produk dan outline STP, serta informasi yang telah dikumpulkan di iterasi sebelumnya, buatlah 4 query untuk Tavily API yang berfokus pada informasi yang masih kurang. Tavily API adalah layanan pencarian informasi melalui web browser yang dapat menjawab berbagai topik dari product_description yang bersifat umum hingga spesifik.

        Instruksi:
        1. Hanya buat query untuk bagian outline STP yang belum terisi atau belum lengkap berdasarkan hasil dari iterasi sebelumnya.
        2. Buat Query dengan bahasa Indonesia dan juga Inggris.
        3. Buat query dengan gaya dan kedalaman yang berbeda, mencakup aspek umum dan spesifik sesuai kebutuhan outline.
        4. Jangan ambil query yang sama dengan yang sudah dicari.

        Output format:
        - Hasilkan output dalam format JSON list yang benar-benar satu baris, tanpa baris baru, spasi tambahan, atau indentasi.
        - Format yang diinginkan: ["query_1", "query_2", "query_3", "query_4"]

        Contoh format yang benar:
        ["Analisis demografi konsumen pakaian olahraga premium di negara-negara ASEAN", "Geographic analysis of regions in ASEAN with high interest in sports and active lifestyle", "Tren pertumbuhan industri pakaian olahraga di ASEAN", "Strategi komunikasi pemasaran efektif untuk pakaian olahraga di pasar ASEAN"]

        Ulangi, hasilkan hanya output dalam format list JSON satu baris seperti di atas.
        <END_PROMPT>
        """

        tavily_planner_prompt = PromptTemplate(
            input_variables=["product_description", "outline", "context", "iteration_count", "searched_query"],
            template=tavily_planner_prompt_template
        )
        tavily_planner_chain = tavily_planner_prompt | llm | StrOutputParser()

        # Grader Chain
        grader_prompt_template = """
        <START_PROMPT>
        Deskripsi Produk: {product_description}
        Outline STP: {outline}
        Informasi yang Terkumpul: {context}

        Tugas Anda adalah menilai apakah informasi yang terkumpul dalam context sudah memenuhi semua poin yang ada di outline STP. Periksa tiap bagian dari outline STP dan pastikan apakah informasi yang dibutuhkan sudah lengkap atau masih ada yang kurang.

        Jika semua poin di outline STP sudah terisi dengan informasi yang sesuai, beri hasil "false". Jika masih ada poin yang kurang atau belum lengkap, beri hasil "true".

        Jawaban dalam format String hanya bisa:
        true atau false
        PERHATIKAN hanya boleh true atau false saja pastikan huruf kecil
        <END_PROMPT>
        """

        grader_prompt = PromptTemplate(
            input_variables=["product_description", "outline", "context"],
            template=grader_prompt_template
        )
        grader_chain = grader_prompt | llm | StrOutputParser()

        # Writer Chain
        writer_prompt_template = """
        <START_PROMPT>
        Deskripsi Produk: {product_description}
        Outline STP: {outline}
        Informasi Terkumpul (Context): {context}

        Tugas Anda adalah menuliskan laporan akhir berdasarkan outline STP yang sudah diberikan. Anda perlu menganalisis Deskripsi Produk, Outline STP, dan Context yang sudah terkumpul untuk menulis laporan. Ikuti instruksi berikut. Informasi ini akan digunakan untuk membantu Ekspor ke Luar Negeri:

        1. Tulis laporan dalam format paragraf untuk setiap bagian outline STP, tanpa menggunakan bullet point.
        2. Saat mengambil informasi dari Context, sertakan referensi atau sumbernya, tulis dalam hyperlink.
        3. Anda dapat menambahkan analisis tambahan berdasarkan wawasan Anda sendiri, asalkan tetap relevan dan selaras dengan Context yang diberikan.
        4. Pastikan setiap bagian dari laporan sesuai dengan struktur outline STP yang diberikan.
        5. Anda harus menulis dalam markdown.
        6. Jangan pernah sarankan negara Indonesia, karena ini adalah ekspor.

        <END_PROMPT>
        """

        writer_prompt = PromptTemplate(
            input_variables=["product_description", "outline", "context"],
            template=writer_prompt_template
        )
        writer_chain = writer_prompt | llm | StrOutputParser()

        # Definisi workflow
        workflow = StateGraph(GraphState)

        workflow.add_node("planner", planner)
        workflow.add_node("web_planner", web_planner)
        workflow.add_node("web_retriever", web_retriever)
        workflow.add_node("grader", grader)
        workflow.add_node("writer", writer)

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
        app = workflow.compile()

        # Menjalankan workflow
        inputs = {"product_description": product_description}
        for output in app.stream(inputs):
            pass  # Proses output jika diperlukan
