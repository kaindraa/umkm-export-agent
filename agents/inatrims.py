from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import ast, re
from IPython.display import Markdown, display

def create_path_planner():
    """Creates and returns the path planning chain"""
    prompt_template = """
    Anda adalah agen yang bertugas menentukan path file yang sesuai berdasarkan Query. Struktur folder saya adalah:
    r"C:\\Users\\ASUS\\Downloads\\hackathon-ai\\gov-ai\\inatrims-docs\\<nama negara>\\<nama produk>.txt"

    Negara yang tersedia hanyalah Malaysia dan Produk yang tersedia adalah:
    alas-kaki
    baterai
    hasil-hutan-kayu
    kakao
    kelapa
    kopi
    kosmetik
    lampu-led
    mainan-anak
    mesin-and-peralatan
    minyak-atsiri
    minyak-sawit
    obat-herbal
    pangan-olahan
    peralatan-listrik-and-elektronik
    peralatan-medis
    produk-karet
    produk-komponen-otomotif
    produk-perikanan
    produk-tekstil

    Instruksi:
    - Jika Query mencakup suatu negara tetapi tidak menyebutkan produk tertentu, berikan semua path file untuk negara tersebut.
    - Jika Query mencakup negara dan produk, berikan hanya path yang sesuai dengan negara dan produk yang diminta.
    - Jika query mencakup produk secara implisit atau sinonim, identifikasi dan berikan path yang relevan.
    - Jika tidak ada negara atau produk tersebut di itu maka hanya keluarkan list kosong.
    - Pastikan bahwa jawaban Anda mempertimbangkan hubungan implisit yang mungkin ada antara query dan context.
    - Anda harus menuliskan query dalam 1 baris saja berupa format yang sudah ditentukan tanpa
    - Contoh yang salah ```plaintext
[]
```
    - Contoh yang benar ["C:\\Users\\ASUS\\Downloads\\hackathon-ai\\gov-ai\\inatrims-docs\\malaysia\\produk-tekstil.txt"]
    - Anda tidak perlu memberi formatting apapun, pastikan anda tulis 1 baris saja, ini sangat penting untuk karir saya

    Query: {query}
    Jawaban: 
    """
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template
    )
    
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    return prompt | llm | StrOutputParser()

def create_content_writer():
    """Creates and returns the content writing chain"""
    prompt_template = """
    Context: {context}

    Deskripsi: Context ini berisi informasi tentang regulasi dan standar teknis ekspor.
    Anda adalah seorang agen yang bertugas menjawab pertanyaan pengguna tentang Regulasi dan Standar Mutu ekspor ke luar negeri.
    Berdasarkan query berikut: "{query}", gunakan informasi dari context untuk menjawab pertanyaan.

    Jika jawaban terdapat dalam context, tulis jawaban tersebut.
    Jika jawaban mengacu pada sumber eksternal dalam context, sebutkan bahwa jawabannya dapat ditemukan di link tersebut.
    Jika jawaban tidak ditemukan dalam context, tulis "Jawaban tidak ditemukan dalam context. Untuk informasi lebih lanjut, silakan kunjungi sumber resmi seperti Kementerian Perindustrian Malaysia atau Badan Kelapa Sawit Malaysia (Malaysian Palm Oil Board/MPOB)."

    Tulis jawaban dalam format Markdown.
    """
    
    prompt = PromptTemplate(
        input_variables=["paths", "context", "query"],
        template=prompt_template
    )
    
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    return prompt | llm | StrOutputParser()

def retrieve_content(paths):
    """Retrieves content from the specified file paths"""
    context = "Kosong"
    if paths:
        context = ""
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    context += file.read() + "\n"
            except FileNotFoundError:
                context += f"File tidak ditemukan: {path}\n"
            except Exception as e:
                context += f"Kesalahan membaca file {path}: {e}\n"
    return context

class InatrimsProcessor:
    """Main class to handle INATRIMS query processing"""
    def __init__(self):
        self.path_planner = create_path_planner()
        self.content_writer = create_content_writer()
    
    def process_query(self, query):
        """
        Process a query and return the formatted response
        
        Args:
            query (str): The query to process
            
        Returns:
            str: Markdown formatted response
        """
        try:
            # Get relevant file paths
            print(query)
            paths_str = self.path_planner.invoke({"query": query})
            print(paths_str)
            paths_str = paths_str.replace('\\', '\\\\')
            paths = ast.literal_eval(paths_str)
            # print('alif ganteng')
            # Retrieve content from files
            context = retrieve_content(paths)
            
            # Generate response
            print(context)
            result = self.content_writer.invoke({
                "query": query,
                "paths": paths,
                "context": context
            })
            
            return result
        except Exception as e:
            return f"Error processing query: {str(e)}"

def inatrims(query):
    """
    Main function to process INATRIMS queries
    
    Args:
        query (str): The query to process
        
    Returns:
        None: Displays the formatted response using IPython.display
    """
    processor = InatrimsProcessor()
    result = processor.process_query(query)
    # display(Markdown(result))
    print(result)



query = input('Masukkan pertanyaan Anda: ')
inatrims(query)