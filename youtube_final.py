import os
from dotenv import load_dotenv
import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
 
# Load environment variables
load_dotenv()
 
# Ensure API key is loaded correctly
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing!")
 
# Set the API key for the OpenAI Python client
openai.api_key = api_key
 
app = FastAPI()
 
# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="D:/GenAI/chroma_db")
collection = chroma_client.get_or_create_collection(name="youtube_video_transcripts")
 
# Recursive text splitting function using LangChain's RecursiveCharacterTextSplitter
def recursive_text_splitter(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks
 
# Function to create embeddings for each chunk using OpenAI's embedding model
def create_embeddings(chunks: list):
    chunk_embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            model="text-embedding-3-large",
            input=chunk
        )
        embeddings = response['data'][0]['embedding']
        chunk_embeddings.append(embeddings)
   
    return chunk_embeddings
 
# Function to add embeddings to ChromaDB
def add_embeddings_to_chromadb(video_id: str, chunks: list, embeddings: list):
    for i, chunk in enumerate(chunks):
        collection.upsert(
            documents=[chunk],
            ids=[f"{video_id}_{i}"],
            embeddings=[embeddings[i]]
        )
 
# Function to search ChromaDB for relevant content based on embeddings
def search_chromadb_by_embedding(query: str):
    query_embedding = openai.Embedding.create(
        model="text-embedding-3-large",
        input=query
    )['data'][0]['embedding']
   
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
 
    # Extract relevant documents
    relevant_documents = results.get("documents", [])
    flattened_docs = [item for sublist in relevant_documents for item in sublist]
    relevant_text = flattened_docs  # Keep it as a list instead of a single string
    print("Relevant text:", type(relevant_text))
    
    # Print the relevant text
    print("Relevant text:", relevant_text)
   
    return relevant_text

 
# Use OpenAI's ChatCompletion for answering the question
def answer_question_from_video_text(video_text: list, user_question: str, llm_model: str = "gpt-4o-mini"):
    # Join the list of relevant text into a single string
    combined_video_text = " ".join(video_text)
    
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are given a transcript of a YouTube video. Answer the following question based on the video content."},
            {"role": "system", "content": combined_video_text},
            {"role": "user", "content": user_question}
        ]
    )
   
    answer = response['choices'][0]['message']['content']
   
    return answer

 
# First API: Upload YouTube video content and store in ChromaDB
@app.post("/upload-youtube-content/")
async def upload_youtube_content(request: Request):
    data = await request.json()
    youtube_link = data.get("youtube_link")
 
    if not youtube_link:
        return JSONResponse(status_code=400, content={"error": "YouTube link is missing."})
 
    try:
        # Step 1: Load the video transcript using YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(youtube_link)
        video_documents = loader.load()
 
        # Combine the documents (video transcript) into a single text string
        video_text = " ".join([doc.page_content for doc in video_documents])
 
        # Step 2: Chunk the video content using recursive text splitter
        chunks = recursive_text_splitter(video_text)
 
        # Step 3: Create embeddings for each chunk
        embeddings = create_embeddings(chunks)
 
        # Step 4: Add the chunks and their embeddings to ChromaDB
        video_id = youtube_link.split("=")[-1]
        add_embeddings_to_chromadb(video_id, chunks, embeddings)
 
        return JSONResponse(content={"message": "Video content uploaded successfully."})
 
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An internal error occurred: {str(e)}"}
        )
 
# Second API: Answer questions based on the stored YouTube video content
@app.post("/answer-from-youtube/")
async def answer_from_youtube(request: Request):
    data = await request.json()
    user_question = data.get("question")
 
    if not user_question:
        return JSONResponse(status_code=400, content={"error": "Question is missing."})
 
    try:
        # Perform a similarity search in ChromaDB using the user question
        relevant_content = search_chromadb_by_embedding(user_question)
 
        # Use LLM to answer the question based on the relevant content from ChromaDB
        answer = answer_question_from_video_text(relevant_content, user_question)
 
        return JSONResponse(content={"answer": answer, "relevant_answers": relevant_content})
 
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An internal error occurred: {str(e)}"}
        )
 
# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)