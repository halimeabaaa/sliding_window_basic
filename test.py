from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


pdf_loader=PyPDFLoader("dosya_adi.pdf")
docs=pdf_loader.load()

def sliding_window_summarize(docs):
    text="\n".join(doc.page_content for doc in docs)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        separators=["\n\n","\n",".","!","?"," "]
    )
    chunks=splitter.split_text(text)
    chunks = chunks[:14]
    print(f"toplam chunk sayısı: {len(chunks)}")

    model=ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",temperature=0.1)

    prompt_chain=load_summarize_chain(model,chain_type="map_reduce")

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"\n Window {i+1} özetleniyor...")

        doc = Document(page_content=chunk)

        summary = prompt_chain.invoke([doc])
        summaries.append(summary)


    return summaries



if __name__ == "__main__":

    ozet=sliding_window_summarize(docs)

    print("\n\n----- Tüm Pencerelerin Özetleri -----")
    for i, ozet in (ozet):
        print(f"Pencere {i+1} Özeti:\n{ozet}\n")





