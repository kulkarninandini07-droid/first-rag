from groq import Groq

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Say hello to Nandini who is building a RAG chatbot!"}
    ]
)

print(response.choices[0].message.content)