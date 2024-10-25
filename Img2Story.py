import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from nlp_utils import clean_sentence
import os
import pickle
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Set device for PyTorch
device = torch.device('cpu')

# Load necessary models and vocabulary
embed_size = 256  # Assuming it's the same as during training
hidden_size = 512  # Assuming it's the same as during training
vocab_file = "vocab.pkl"  # Assuming the name of the vocabulary file
encoder_file = "encoder-10.pkl"  # Update with the appropriate file name
decoder_file = "decoder-10.pkl"  # Update with the appropriate file name

# Load the vocabulary
with open(os.path.join("/Users/chaiganeshj/PycharmProjects/pythonProject1/", vocab_file), "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Load the trained model weights
encoder.load_state_dict(torch.load(os.path.join("./model", encoder_file), map_location=device))
decoder.load_state_dict(torch.load(os.path.join("./model", decoder_file), map_location=device))

# Move models to the appropriate device
encoder.to(device)
decoder.to(device)

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# Image preprocessing function
def preprocess_image(image):
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = transform_test(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to generate caption from the uploaded image
def generate_caption(image):
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        features = encoder(preprocessed_image).unsqueeze(1)
        output = decoder.sample(features)
    caption = clean_sentence(output, vocab.idx2word)
    return caption

# Function to generate a story
def generate_story(input_text, no_words):
    llm = Ollama(model='llama2')
    template = """Write a story regarding {input_text} within {no_words} words and provide a suitable title and moral."""
    prompt = PromptTemplate(input_variables=["input_text", "no_words"], template=template)
    formatted_prompt = prompt.format(input_text=input_text, no_words=no_words)
    response = llm(formatted_prompt)
    return response

# Function to generate social media content
def generate_social_media_content(input_text, no_words):
    llm = Ollama(model='llama2')
    template = """Create a social media post regarding {input_text} within {no_words} words. Make it engaging and suitable for sharing."""
    prompt = PromptTemplate(input_variables=["input_text", "no_words"], template=template)
    formatted_prompt = prompt.format(input_text=input_text, no_words=no_words)
    response = llm(formatted_prompt)
    return response

# Function to generate advertisement idea
def generate_advertisement_idea(input_text, no_words, ad_type, purpose):
    llm = Ollama(model='llama2')
    template = """Develop an advertisement idea for a {ad_type} regarding {input_text} within {no_words} words. 
    The purpose is to {purpose}. Make it creative and appealing."""
    prompt = PromptTemplate(input_variables=["input_text", "no_words", "ad_type", "purpose"], template=template)
    formatted_prompt = prompt.format(input_text=input_text, no_words=no_words, ad_type=ad_type, purpose=purpose)
    response = llm(formatted_prompt)
    return response

# Streamlit UI
st.set_page_config(page_title="Image to Content Generator", page_icon='📚', layout='centered')

st.header("Image to Content Generator 📚")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption for the uploaded image
    caption = generate_caption(image)
    st.subheader(f"Generated Caption: {caption}")

    # Select the type of content to generate
    content_type = st.selectbox("Choose the type of content to generate", 
                                ("Story", "Social Media Content", "Advertisement Idea"))

    # Fields for content generation
    no_words = st.text_input("No of words")
    
    if content_type == "Advertisement Idea":
        ad_type = st.selectbox("Type of Advertisement", ["Product", "Service", "Event"])
        purpose = st.text_input("Purpose of the Advertisement (e.g., promote product, attract clients)")

    submit_content = st.button("Generate Content")

    if submit_content:
        # Generate content based on the selected type
        if content_type == "Story":
            result = generate_story(caption, no_words)
        elif content_type == "Social Media Content":
            result = generate_social_media_content(caption, no_words)
        elif content_type == "Advertisement Idea":
            result = generate_advertisement_idea(caption, no_words, ad_type, purpose)
        
        st.write(result)