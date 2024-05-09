import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.llms import CTransformers
from get_embedding_function import get_embedding_function
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

MODEL_FILE = "/home/huy/vinallama-7b-chat-GGUF/vinallama-7b-chat_q5_0.gguf" 

PROMPT_TEMPLATE = """
<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố gắng tạo ra một câu trả lời. Giữ câu trả lời ngắn gọn nhất có thể:
{context}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

def main():
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    query_text = "Cách thực hiện cấp giấy phép nhập khẩu xuất bản phẩm không kinh doanh?"
    res = query_rag(query_text)
    print(res)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # results = db.similarity_search_with_score(query_text, k=3)

    # context_text = "\n---\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    # # print(prompt)
    

    compressor = FlashrankRerank()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.invoke(query_text)
    # print(compressed_docs[0])

    context_text = "\n---\n".join([doc.page_content for doc in compressed_docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # docs = retriever.invoke(query_text)
    # print(docs)

    model = Ollama(model="vina7b")
    # # Load LLM
    # model = CTransformers(
    #         model=MODEL_FILE,
    #         model_type="llama",
    #         max_new_tokens=1024,
    #         temperature=0.05
    #     )

    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    return response_text #response_text


if __name__ == "__main__":
    main()
