from pypdf import PdfReader # used to extract text from pdf
from langchain.text_splitter import CharacterTextSplitter # split text in smaller snippets
#import os # read API key from environment variables. Not required if you are specifying the key in notebook.
from openai import OpenAI # used to access openai api
import json # used to create a json to store snippets and embeddings
from numpy import dot # used to match user questions with snippets.
import openai
from app_utils import *
import os

#max relevant images to display
MAX_IMAGES = 1

#path of the folder containing the pdf files
folder_path="."

#find and read all pdf files into a text file
load_pdf(folder_path)

#create embeddings and write to json file
create_embeddings()

import streamlit as st
st.title("Pdf Query Bot")

# Initialize chat history and prompt counter
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if os.path.exists(message["content"]):
            st.image(message["content"])
        else:
            st.markdown(message["content"])

# Check if the user has reached the prompt limit
if st.session_state.prompt_count >= 5:
    st.markdown("You have reached the maximum number of prompts allowed.")
else:
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Increment the prompt counter
        st.session_state.prompt_count += 1

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = answer_users_question(prompt)
        content = response[0]
        meta = response[1]

        imgs = []
        for data in meta:
            if "images" in data:
                images = data['images']
                for i in images:
                    if len(imgs) < MAX_IMAGES:
                        imgs.append(i)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response[0])
            if (response[0] != "Sorry! I can't help you.") & (len(response[0]) > 30):
                if imgs:
                    for i in imgs:
                        st.image(i)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", 
                                          "content": f"{response[0]}"})

        if len(imgs) > 0 & (response[0] != "Sorry! I can't help you.") & (len(response[0]) > 30):
            for i in imgs:
                st.session_state.messages.append({"role": "assistant", 
                                                  "content": f"{i}"})
