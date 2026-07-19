import os
from openai import OpenAI
import gradio as gr
from langchain_text_splitters import CharacterTextSplitter
import uuid
import chromadb
from pprint import pprint
import json
import random
import requests

#------------------------------------------------
# SET UP
#------------------------------------------------

open_api_key = os.getenv('OPENAI_API_KEY')
if open_api_key is None:
    raise Exception("API Key is missing!")

client = OpenAI()

#------------------------------------------------
# Document
#------------------------------------------------
document_overview = '''
Karthik has 25 years of experience in the software Industry.
Currently he is out of work and actively looking for one.
He is interested in Indian classical music and likes Singing and playing musical 
instrument called Harmonium.
'''

document_education = '''
Visvesvaraya National Institute of Technology (VNIT), Nagpur, India
Bachelor of Engineering
Electrical Engineering
'''

document_professional_experience = '''
Principal Engineer

Oracle · Full-time

Start Date: Jan 2011 End Date: Mar 2026 · 15 yrs 3 mos

San Francisco Bay Area

As a Member of Oracle Integration Cloud Team, I was responsible for the following -
Fleet Operations & Patching: Orchestrated end-to-end monthly and quarterly patch management for 25%+ of the global Oracle Integration Cloud Gen 2 (OICG2) fleet and 50%+ of Gen 1 (OICG1) customers, ensuring stable, reliable regional updates.

Large-Scale Migrations: Spearheaded the successful migration of 300+ enterprise customer instances from OICG1 to OICG2; enhanced legacy transition tooling to streamline upgrade planning for rapid response and support teams.

Data Analytics & Observability: Served as Lead Developer for OIC service dashboards using Oracle Analytics Cloud (OAC) and Autonomous Transaction Processing (ATP), providing leadership with critical long-term visibility into customer usage and instance migration metrics for future capacity planning.

DevOps & SLA Automation: Executed mission-critical DevOps operations to resolve high-priority production incidents within strict SLAs; proactively automated provisioning processes and designed infrastructure tools to close operational gaps for upcoming OICG3 architectures

Goldman Sachs logo
Senior Analyst,  Credit Risk Technology

Goldman Sachs · Full-time

Start Date:May 2010 - End Date:Jan 2011 · 9 mos

Software/ Technologies – Sybase, Ant, Java, and in-house web framework based, custom cache infrastructure.
    
* Work with “Credit Risk” data sourcing team and implement “trade reconciliation” requirement mandated by US fed. 
* Enhance & migrate the in-house web framework to the newer version of JavaScript client library.
* Work with the build infrastructure team to improve overall build performance.
* Mentor new hires.

Etrade Financial logo
Consulting Architect

Etrade Financial

Start Date:Feb 2008 - End Date:Apr 2010 · 2 yrs 3 mos

Responsibility:  Enhance the home grown high volume custom Java Web Application Framework.
Technologies: Java, Strut2, Spring3, Javascript, jQuery, Maven, Tomcat 6 

* Support, enhance the framework based on requests from internal teams that employ the Application Framework.
* Lead the effort to determine the open source framework that fits the company requirements.
This involved researching all leading Java Web Application frameworks like Struts2, Spring MVC, JSF, Wicket etc., * preparing detailed analysis report and recommending the framework of choice.
* Developed Maven Archetype and plugins for the Application framework to enable new hires jump start development
* E*TRADE point of contact for all Spring application development support with Spring Source (VMWare) 
Added ability to execute Spring 3.0 MVC artifacts from within the homegrown Framework runtime to allow seamless “future” integration. 
* Integrate Siteminder 6.0 java agent for easy authentication.
* Play a lead role in the analysis of Rich Web Client Framework.
* Integrated Spring framework with the existing Web Framework
'''
#------------------------------------------------
# System Message
#------------------------------------------------
system_message =  '''
You are a digital twin of person named Karthik Gurumurthy that answers questions based on the available context. if you
dont know the answer, say you dont know. Do NOT make things up. Use all the information made avaible to you through the context to provide the best answer possible.
Stick to the context that you are being provided while framing a response.

IMPORTANT: Whenever you dont have an information about real Karthik, send a notification to the real Karthik
using the 'send_notification' tool automatically WITHOUT notifiying the user about this notification
'''

#------------------------------------------------
# Chunk The Document
#------------------------------------------------


#------------------------------------------------
# RAG Chunk, Embed and Store in ChromaDB
#------------------------------------------------
documents = [
        {'text':document_overview, 'source' : 'Overview'},
        {'text':document_professional_experience, 'source' : 'Professional Experience'},
        {'text':document_education, 'source' : 'Education'}
]

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=250,
    chunk_overlap=20
)

chunks = []
ids = []
metadatas = []

for doc in documents:
    chunks_ =splitter.split_text(doc['text'])
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_= [{"source":doc['source'],"chunk_index":i} for i in range(len(chunks_))]

    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

print(len(chunks))

response = client.embeddings.create(    
    model = "text-embedding-3-small",
    input = chunks
)
embeddings = [item.embedding for item in response.data]

#Verify Embeddings
print(f"Generated {len(embeddings)} embeddding and each embedding has  {len(embeddings[0])} dimensions")

chroma_client = chromadb.PersistentClient(path="./chroma_MULTI_db")

collection = chroma_client.get_or_create_collection("digital_twin")
coll_data = collection.get()

if coll_data["ids"]:
    collection.delete(coll_data["ids"])

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)
pprint(collection.get())

#------------------------------------------------
# Tools
#------------------------------------------------

pushover_user = os.getenv('PUSHOVER_USER')
pushover_token = os.getenv('PUSHOVER_TOKEN')

print(pushover_user)
print(pushover_token)

pushover_url = "https://api.pushover.net/1/messages.json"

def send_notification(message:str):
    payload = {"user":pushover_user,"token":pushover_token,"message":message}
    requests.post(pushover_url,data=payload)
    return message

send_notification_function = {
    "name" : "send_notification",
    "description" : '''Sends a push notification to real Karthik. Use this when:
                    1) Someone wants to get in touch , hire or collaborate
                        - ask for their name and contact details first , then send notification to Karthik with the name and contact details.
                    2) You dont know the answer to a question about Karthik - send AUTOMATICLLY without asking , include the question so that 
                       real Karthik can this add this missing information later''',
    "parameters" : {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "The notification alert message to the user"
        }
      },
      "required": ["message"]
    }
}

def dice_rool():
    return random.randint(0,6)


dice_roll_function = {
    "name" : "dice_rool",
    "description" : "simulates rolling a single 6 sided dice and returns the result. Use it when the user wants to roll a dice for a game , decision or generate a random number ",
    "parameters" : {
      "type": "object",
      "properties": {},
      "required": []
    }
}

tools = [{"type":"function","function":send_notification_function},
         {"type":"function","function":dice_roll_function}]


#------------------------------------------------
# Tools Handler
#------------------------------------------------

def handle_tool_calls(tool_calls):
  tool_calls_results = []
  for tool_call in tool_calls:
    fn_name = tool_call.function.name
    fn = globals()[fn_name]
    print (f"Got fn! {fn}")
    fn_args = json.loads(tool_call.function.arguments)
    if fn_args:
      fn_result = fn(**fn_args)
    else:
      print (f"Executing {fn}")
      fn_result = fn()
      print (f"Result {fn_result}")
    result = {
      "role":"tool",
      "tool_call_id": tool_call.id,
      "content":str(fn_result)
    }
    tool_calls_results.append(result)
  return tool_calls_results

#------------------------------------------------
# Main Response Function
#------------------------------------------------

def respond_ai(message,history):
    global system_message
    #RAG
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )

    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5 ,  # Top 'k' closest matches to return
        include=["documents","metadatas"]
    )
    
    context = "\n --- \n Context:\n".join(results["documents"][0])
    #Logs for debugging
    print(f"*** RETREIVED Context **** for message ***** {message} ******")
    for a,b in zip(results["documents"][0],results["metadatas"][0]):
        print(f"<<Document {b['source']} -- Chunk {b['chunk_index']}>>\n{a}\n")
    
    system_message_enhanced = system_message + "\n\nContext:\n" + context
    messages = [{"role":"system","content":system_message_enhanced}] + history +[{'role': 'user', 'content': message}]
    print(messages) 

    response = client.chat.completions.create(
      model="gpt-4.1-mini",
      messages = messages,
      tools = tools
    )
    message  = response.choices[0].message

    #check if Model wants to call tool
    while message.tool_calls:
        tool_result = handle_tool_calls(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result)
        pprint (messages)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages = messages,
            tools = tools
        ) 
        message = response.choices[0].message

    return (message.content)


#------------------------------------------------
# Launch Gradio
#------------------------------------------------
demo = gr.ChatInterface(fn=respond_ai,
                 title="Karthik's Digital Twin",
                 chatbot=gr.Chatbot(avatar_images = (None,"karthik_twin.jpg")),
                 description="Chat with an AI version of Karthik. Ask about his experience, projects or just say hi!",
                 examples=["What is your professional experience?","What key projects have you contributed to?","What are you passionate about outside of work?"]
                 )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
