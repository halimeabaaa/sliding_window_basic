from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()

text=""" Metninizi giriniz... """

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n","\n\n",".","?","!"," "]
)
chunks=text_splitter.create_documents([text])#create_documents list alır
chunks=chunks[:14]
llm=ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

def sliding_window_summarize(chunks):
    summaries=[] #her chunk özeti burada toplanacak
    for i,doc in enumerate(chunks): #her chunk doc içine alınır
        chain=load_summarize_chain(llm,chain_type="stuff")
        summary=chain.run([doc])
        print(f"window {i+1} özetleniyor--------")
        summaries.append(summary)
    return summaries
summaries=sliding_window_summarize(chunks)
sum="\n".join(summaries)

sumarry_doc=[Document(page_content=sum)]


prompt_template=ChatPromptTemplate.from_messages(
    [("system","""
Sen bir metin analiz yardımcısısın.Yukarıda sana verilen texte dayanarak çalışırsın. 
{context}
Kullanıcının amacı text  hakkında bilgi edinmek, özet çıkarmak ve metne dayalı sorulara cevap almaktır.
Kullanıcının sorularını verilen text üzerinden anlamlı ve açık şekilde yanıtlamak.
Eğer kullanıcı sadece "özetle" veya "bu metni özetle" gibi bir komut verirse, tüm metnin özetini sunmak.
Sorulara sadece özetten elde edilebilecek bilgiler doğrultusunda yanıt ver.
Eğer özetten çıkarılamayan bir bilgi istenirse, "Bilmiyorum".
"""),
    ("human","{input}")]


)

chain_text=create_stuff_documents_chain(llm,prompt_template)

query="metni özetle"
response=chain_text.invoke({"input":query,"context":sumarry_doc})
print(response)
