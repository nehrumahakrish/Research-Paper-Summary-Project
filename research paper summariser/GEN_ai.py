import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.document_loaders import PyPDFLoader

# Load the model and tokenizer from Hugging Face
model_name = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Function to summarize the PDF content based on word count
def summarize_pdf(pdf_file_path, num_words):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    
    # Combine all pages' content into a single string
    full_text = " ".join([doc.page_content for doc in docs])

    # Approximate the number of tokens equivalent to words
    tokens_per_word = 1.33  # Estimated value (varies depending on the model)
    max_length = int(num_words * tokens_per_word)

    # Summarize the full text with the given word limit
    summary = summarizer(full_text, max_length=max_length, min_length=int(max_length*0.7), do_sample=False)

    return summary[0]['summary_text'], full_text  # Return full text for further refinement

# Function to refine the summary based on a custom prompt
def refine_summary(full_text, prompt, num_words):
    prompt_text = f"Focus on: {prompt}\n{full_text}"

    # Approximate the number of tokens equivalent to words
    tokens_per_word = 1.33  # Estimated value (varies depending on the model)
    max_length = int(num_words * tokens_per_word)

    # Summarize the full text with the given word limit, focusing on the prompt
    refined_summary = summarizer(prompt_text, max_length=max_length, min_length=int(max_length*0.7), do_sample=False)

    return refined_summary[0]['summary_text']

# Streamlit user interface
st.title("Research Paper Summarizer by NEHRU")

# Step 1: File uploader in Streamlit
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Step 2: Ask for word count for the initial summary
num_words = st.number_input("How many words do you want in the summary?", min_value=50, max_value=1000, value=200)

# Variable to store full text of the PDF
full_text = ""

# Display the summary if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded PDF temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("File successfully uploaded. Generating summary...")

    # Call the function to summarize the uploaded PDF
    summary, full_text = summarize_pdf("uploaded_pdf.pdf", num_words)

    # Display the summary in Streamlit
    st.subheader(f"Summary ({num_words} words):")
    st.write(summary)

    # Step 3: Show prompt box after initial summary
    st.subheader("Refine the summary with a custom prompt")
    custom_prompt = st.text_input("Enter a custom prompt to focus the summary (optional):")

    if custom_prompt:
        st.write(f"Refining summary based on prompt: '{custom_prompt}'...")

        # Refine the summary based on the custom prompt
        refined_summary = refine_summary(full_text, custom_prompt, num_words)

        # Display the refined summary
        st.subheader(f"Refined Summary ({num_words} words):")
        st.write(refined_summary)
