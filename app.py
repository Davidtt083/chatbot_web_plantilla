from flask import Flask, render_template, request, session, jsonify
import google.generativeai as genai
import os
from gemini.promts import instruccion2,documents
import google.generativeai as genai
import tempfile
import pandas as pd
import numpy as np
from IPython.display import display
from IPython.display import Markdown
import textwrap
import markdown
from markupsafe import Markup
import re

generation_config = {
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 800,
}

safety_settings = [
     {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
  "input: ",
  "output: ",
]
model_embedding = 'models/embedding-001'
chat = model.start_chat(history=[])
conversations = []
instruction = instruccion2
AUDIO_FOLDER = 'templates/audio_files'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="Aquí_va_tu_API_Key")

def clean_text(text):
    """
    Función para limpiar el texto y eliminar las etiquetas HTML.
    """
    # Eliminar las etiquetas HTML utilizando expresiones regulares
    cleaned_text = re.sub(r'<[^>]+>|\*|#', '', text)
    return cleaned_text

app = Flask(__name__, static_folder='templates')
conversations = []
app.secret_key = "1"


@app.route('/', methods=['GET', 'POST'])
def home():
    if 'conversations' not in session:
        session['conversations'] = []

    if request.method == 'GET':
        return render_template('index.html', conversations=session['conversations'])

    elif request.method == 'POST':

        df = pd.DataFrame(documents)
        df.columns = ['Title', 'Text']


        def embed_fn(title, text):
            return genai.embed_content(model=model_embedding,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

        df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)

        query = request.form['question']

        request_e = genai.embed_content(model=model_embedding,
                              content=query,
                              task_type="retrieval_query")

        def find_best_passage(query, dataframe):
            """
            Compute the distances between the query and each document in the dataframe
            using the dot product.
            """
            query_embedding = genai.embed_content(model=model_embedding,
                                        content=query,
                                        task_type="retrieval_query")
            dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
            idx = np.argmax(dot_products)
            return dataframe.iloc[idx]['Text'] # Return text from index with max value

        passage = find_best_passage(query, df)
        print(passage)

        question = request.form['question']

        full_prompt = f"{instruction}\\n{passage}"

        response = chat.send_message(question + f"""{instruction}\n\n{passage}""")

        # Imprimir los tokens de entrada y salida
        print("Input tokens:", len(full_prompt.split()))
        print("Output tokens:", len(response.text.split()))

        bot_response = response.text
        print(full_prompt)

        response_lines = [Markup(line.replace('**', '<b>').replace('**', '</b>').replace('*', '<li>')) for line in bot_response.split('\n') if line.strip()]

        session['conversations'].append({'user': question, 'bot': response_lines})




        return jsonify({'response': response_lines})



if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 4000)))