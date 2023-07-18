import os
import uuid

from flask import Flask, request, jsonify, make_response

from langchain.chains import ConversationChain
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Milvus

from pymilvus import db, connections, utility

chat_sessions = {}

MILVUS_IP="127.0.0.1"

connections.connect(host=MILVUS_IP, port=19530)
utility.drop_collection("OnTopic")
utility.drop_collection("OffTopic")

text_embeddings = VertexAIEmbeddings()
palm = ChatVertexAI()

app = Flask(__name__)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    session_id = request.cookies.get('session_id')
    if session_id is None or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ConversationChain(
            llm=palm, memory=ConversationBufferMemory()
        )
    chat = chat_sessions[session_id]

    user_input = request.get_json()['input']
    on_topic_docs = on_topic_db.similarity_search_with_score(user_input, k=128)
    off_topic_docs = off_topic_db.similarity_search_with_score(user_input, k=128)
    print("Closest on-topic docs:", [(doc[0].page_content, doc[1]) for doc in on_topic_docs[0:3]])
    print("Closest off-topic docs:", [(doc[0].page_content, doc[1]) for doc in off_topic_docs[0:3]])
    on_topic_distance = sum([doc[1]**2 for doc in on_topic_docs])
    off_topic_distance = sum([doc[1]**2 for doc in off_topic_docs])
    print("On topic distance", on_topic_distance)
    print("Off topic distance", off_topic_distance)

    response = make_response(jsonify({
        'response': chat.predict(input=user_input) if on_topic_distance < off_topic_distance else "Sorry. This is not a topic that I know."
    }))
    response.set_cookie('session_id', session_id)
    return response

@app.route('/api/embeddings', methods=['POST'])
def embeddings():
    global on_topic_db
    global off_topic_db
    on_topic_examples = [example["question"] for example in request.get_json()['on_topic']]
    off_topic_examples = [example["question"] for example in request.get_json()['off_topic']]
    for i in range(0, len(on_topic_examples), 5):
        on_topic_db = Milvus.from_texts(
            on_topic_examples[i:i+5],
            text_embeddings,
            collection_name="OnTopic",
            connection_args={"host": MILVUS_IP, "port": "19530"},
        )
    for i in range(0, len(off_topic_examples), 5):
        off_topic_db = Milvus.from_texts(
            off_topic_examples[i:i+5],
            text_embeddings,
            collection_name="OffTopic",
            connection_args={"host": MILVUS_IP, "port": "19530"},
        )
    return ""


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)