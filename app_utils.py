#from transformers import pipeline
from pypdf import PdfReader # used to extract text from pdf
from langchain.text_splitter import CharacterTextSplitter # split text in smaller snippets
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI # used to access openai api
import json # used to create a json to store snippets and embeddings
from numpy import dot # used to match user questions with snippets.
import openai
import glob
import streamlit as st
import os
import fitz


EXTRACTED_PDF_TEXT = "pdf_text.json" # text extracted from pdf
EXTRACTED_JSON_PATH = "extracted.json" # snippets and embeddings
OPENAI = st.secrets["OPENAI_KEY"]# replace this with your openai api key or store the api key in env
EMBEDDING_MODEL = "text-embedding-ada-002" # embedding model used

GPT_MODEL = "gpt-3.5-turbo-0125" # gpt model used. alternatively you can use gpt-4 or other models.
CHUNK_SIZE = 1000 # chunk size to create snippets
CHUNK_OVERLAP = 200 # check size to create overlap between snippets
CONFIDENCE_SCORE = 0.75 # specify confidence score to filter search results. [0,1] prefered: 0.75
K=5 #max number of relevant snippets to consider
pdf_description = """ User Guide"""






@st.cache_resource
def load_pdf(folder_path=".",images_folder="images"):

  #get all pdf files
  pdf_files = glob.glob(folder_path+"/*.pdf")

  pdfs_list = []
  
  #iterate over each pdf file
  for pdf_file in pdf_files:

    #name of the pdf file
    pdf_name = os.path.basename(pdf_file)

    #initiate a dictionary to store text from each page
    pages_dict = {"pdf_name":pdf_name, "pages":{},"images":{}}

    #read the pdf file
    with open(pdf_file, 'rb') as f:
        reader = PdfReader(f)

        # Iterate over each page
        for i,page in enumerate(reader.pages):
          if i==0:
            continue

          #extract text
          page_text = page.extract_text()
          
          #code to skip blank pages
          if page_text.strip():
                pages_dict["pages"][f"{i+1}"] = page_text

                

    


    #open pdf using fitz to extract images
    doc = fitz.open(pdf_file)
    exported = []
    xreflist = []
    imglist = []
    #count of pages in the pdf file
    page_count = doc.page_count



    # Check if the image folder exists, if not, create it
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    #iterate through the pages of the doc
    for j,page in enumerate(doc.pages()):
        if j==0:
          continue
      
        # get images in the page
        refs = page.get_images()
        
        images=[]
        for i, ref in enumerate(refs):
            xref, smask, w, h = [ref[n] for n in (0, 1, 2, 3)]
            if xref in exported:
                continue

            #naming the image file
            output = f'{pdf_name}-{page.number + 1:04}-{xref + 1:05}'

            if smask:
                mask = fitz.Pixmap(doc, smask)
                if (mask.width != w) or (mask.height != h):
                    mask = fitz.Pixmap(mask, w, h, None)
                image = fitz.Pixmap(doc, xref)
                image = fitz.Pixmap(image, mask)
                image = fitz.Pixmap(fitz.csRGB, image)
                output = f'{os.path.join(images_folder, output)}.png'
            else:
                image = fitz.Pixmap(doc, xref)
                output = f'{os.path.join(images_folder, output)}.jpg'

            #saving the names of the image files
            images.append(output)

            #saving image to storage
            image.save(output)
            exported.append(xref)

        pages_dict["images"][f"{j+1}"] = images  


    #append pages to pdfs_list
    pdfs_list.append(pages_dict)
  
  
  

  with open(EXTRACTED_PDF_TEXT, "w", encoding="utf-8") as f:
              json.dump(pdfs_list,f)
  
  return




@st.cache_resource
def create_embeddings(file_path=EXTRACTED_PDF_TEXT):

    # Read the content of the file specified by file_path
    with open(EXTRACTED_PDF_TEXT, "r", encoding="utf-8") as f:
            pdfs_list = json.load(f)

    texts = []
    metadata = []
    for pdfs in pdfs_list:
      pages = pdfs["pages"]
      for pg in pages:
        texts.append(pages[pg])

        if pdfs["images"][pg]:
          metadata.append({
              "pdf_name":pdfs["pdf_name"],
              "page_number":pg,
              "images":pdfs["images"][pg]
          })
        else:
          metadata.append({
              "pdf_name":pdfs["pdf_name"],
              "page_number":pg
          })

    
    


    text_splitter = CharacterTextSplitter(separator="\n",
                                            chunk_size=1000,
                                            chunk_overlap=200,
                                            length_function=len)

    #use create documents to split text into snippets with metadata
    documents = text_splitter.create_documents(texts=texts,metadatas=metadata)

    snippets = [document.page_content for document in documents]
    metadatas = [document.metadata for document in documents]

    #Request embeddings using openai for the snippets and the specified model
    embed_model = OpenAIEmbeddings(model=EMBEDDING_MODEL,openai_api_key=OPENAI)

    embeddings = embed_model.embed_documents(snippets)




    
    # Create a JSON object containing embeddings, snippets and metadata
    embedding_json = {
        'embeddings': embeddings,
        'snippets': snippets,
        'metadata':metadatas
    }

    # Convert the JSON object to a formatted JSON string
    json_object = json.dumps(embedding_json, indent=4)

    # Write the JSON string to a file specified by EXTRACTED_JSON_PATH
    with open(EXTRACTED_JSON_PATH, 'w', encoding="utf-8") as f:
        f.write(json_object)


def get_embeddings():

    # Open the JSON file containing embeddings and snippets
    with open(EXTRACTED_JSON_PATH,'r') as file:
        # Load the JSON data into a Python dictionary
        embedding_json = json.load(file)

    # Return the embeddings and snippets from the loaded JSON
    return embedding_json['embeddings'], embedding_json['snippets'],embedding_json["metadata"]



def user_question_embedding_creator(question):


    #Request embeddings using openai for the user question
    embed_model = OpenAIEmbeddings(model=EMBEDDING_MODEL,openai_api_key=OPENAI)

    #get embeddings for 
    embedding = embed_model.embed_query(question)

    
    # Extract and return  the embedding from the API response
    return embedding




def answer_users_question(user_question):

    try:
        # Create an embedding for the user's question
        user_question_embedding = user_question_embedding_creator(user_question)
    except Exception as e:
        # Handle any exception that occurred while using Embedding API.
        return f"An error occurred while creating embedding: {str(e)}"

    embeddings, snippets, metadata = get_embeddings()
    # Calculate cosine similarities between the user's question embedding and the document embeddings
    cosine_similarities = []
    for embedding in embeddings:
        cosine_similarities.append(dot(user_question_embedding,embedding))

    # Pair snippets with their respective cosine similarities and sort them by similarity
    scored_snippets = zip(snippets, cosine_similarities,metadata)
    sorted_snippets = sorted(scored_snippets, key=lambda x: x[1], reverse=True)

    # Filter snippets based on a confidence score and select the top 5 results
    formatted_top_results = [(snipps,meta) for snipps, _score, meta in sorted_snippets if _score > CONFIDENCE_SCORE]
    if len(formatted_top_results) > K:
        formatted_top_results = formatted_top_results[:K]

    top_snippets = [result[0] for result in formatted_top_results]
    top_snippets_meta = [result[1] for result in formatted_top_results]

    # Create the chatbot system using pdf_description provided by the user.
    chatbot_system = f"""You are provided with SEARCH RESULTS from a pdf. This pdf is a {pdf_description}. You need to generate answer to the user's question based on the given SEARCH RESULTS. SEARCH RESULTS as a python list. SEARCH RESULTS and USER's QUESTION are delimited by ``` \n If there is no information available, or question is irrelevent respond with - "Sorry! I can't help you." """

    # Create the prompt using results and user's question.
    prompt = f"""\
    SEARCH RESULTS:
    ```
    {top_snippets}
    ```
    USER'S QUESTION:
    ```
    {user_question}
    ```

    """

    # Prepare the chat conversation and use GPT model for generating a response
    messages = [{'role':'system', 'content':chatbot_system},
                {'role':'user', 'content':prompt}]

    try:
        client = openai.OpenAI(api_key=OPENAI)
        completion = client.chat.completions.create(model=GPT_MODEL,
                                             messages=messages,
                                             temperature=0,
                                             stream=False)
    except Exception as e:
        # Handle exception while communicating with ChatCompletion API
        return f"An error occurred with chatbot: {str(e)}"

    # Return the chatbot response.
    return (completion.choices[0].message.content, top_snippets_meta)
