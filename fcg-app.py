import streamlit as st
import functions as func

file_preparer = func.FilePreparer()

st.title("PDF ChatGenie🧞‍♂️")

# Apply custom styles for chatbox
st.markdown("""
    <style>
        div[data-baseweb="textarea"] > div > textarea {
            font-size: 18px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Check if the API key is already provided
if "openai_api_key" not in st.session_state:

    st.markdown("""
    ### Disclaimer
    Please provide your OpenAI API key to proceed. 
    This key is used for the purpose of this demo and will not be stored or used beyond this session. 
    For safety, it is recommended to reset your API key after using this app.
    """)

    # Prompt for OpenAI API key
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")

    if st.button("Submit"):
        st.session_state.openai_api_key = openai_key
        st.experimental_rerun()

else:
    # Initialize states if not already done
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
        st.session_state.vectorized = False
        st.session_state.chat_mode = False
        st.session_state.chat_started = False

    # File upload and vectorization logic
    if not st.session_state.chat_mode:
        st.write("Upload a PDF file to chat with:")
        file_input = st.file_uploader("Choose a PDF file to chat with")

        if file_input:
            st.success(f"You've selected {file_input.name}.")
            upload_button = st.button("Upload File")

            if upload_button:
                try:
                    file_preparer.upload_to_azure(file_input)
                    st.session_state.uploaded = True
                except ValueError as e:
                    st.error(str(e))

        if st.session_state.uploaded:
            st.success(f"{file_input.name} has been uploaded successfully!")

            # Convert the file to text, split up the text, and vectorize the split-up text
            file_text = file_preparer.convert_PDFfile_to_text(file_input.name)
            split_text = file_preparer.split_the_text(file_text)
            st.session_state.docs = file_preparer.vectorize_text(split_text, st.session_state.openai_api_key)
            st.session_state.vectorized = True
            st.write("File Vectorized successfully!")

        # Only show "Start Chatting" button if the file has been vectorized
        if st.session_state.vectorized:
            start_chat_button = st.button("Start Chatting")
            if start_chat_button:
                st.session_state.chat_mode = True
                st.experimental_rerun()

    # Chat Interface
    if st.session_state.chat_mode:
        # Initialize the chatbot and retrieval system
        chat_bot = func.ChatBot(st.session_state.openai_api_key)
        q = chat_bot.initialize_retrieval_qa(st.session_state.docs)

        # Initialize a chat log if not already done
        if 'chat_log' not in st.session_state:
            st.session_state.chat_log = []

        # Using the session state directly to store the chatbox value
        chatbox = st.text_area("Type your message here...")

        send_message_button = st.button("Send Message")

        if send_message_button and chatbox.strip() != "":
            # Get the chatbot's reply
            return_message = q.run(chatbox)

            # Update the chat log
            st.session_state.chat_log.append({
                "type": "user",
                "message": chatbox
            })
            st.session_state.chat_log.append({
                "type": "bot",
                "message": return_message
            })

        # Display chat log
        for entry in st.session_state.chat_log:
            if entry["type"] == "user":
                st.markdown(f"<div style='font-size: 18px; color: #9e9e9e; margin-bottom: 24px;'>You: {entry['message']}</div>", unsafe_allow_html=True)
            else:
                # Use markdown to style the chatbot's reply with a larger font size
                st.markdown(f"<div style='font-size: 18px; margin-bottom: 24px;'><b>PDF ChatGenie🧞‍♂️:</b> {entry['message']}</div>", unsafe_allow_html=True)
