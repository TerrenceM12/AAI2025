import os
from google import genai
from google.genai import types

# Hardcode your API key for testing
my_api_key = "API_KEY_HERE"
client = genai.Client(api_key=my_api_key)
MODEL_ID = "gemini-2.5-flash"

# --- THE SELF-REFLECTION PROMPT (WITH CRITERIA) ---
reflection_system_instruction = """
You are a highly helpful and accurate general knowledge AI. 
Before you provide a final answer to the user, you MUST draft, critique, and revise your response.

You MUST evaluate your draft against these Strict Criteria:
1. Focus: Does the answer directly address the user's prompt without unnecessary tangents?
2. Length: Is the response concise (ideally under 3-4 short paragraphs)?
3. Format: Is the text easy to scan? (e.g., uses bullet points, bold text, or clear spacing where appropriate).
4. Tone & Audience: Is the language perfectly tailored to the user's requested audience or scenario?
5. Accuracy: Is there any risk of hallucination or confusing phrasing that needs to be clarified?

You MUST format your entire response using these exact headings:

### 1. Initial Draft
[Write your first, unfiltered attempt at answering the user's question.]

### 2. Self-Critique Against Criteria
[Critique your initial draft strictly using the 5 criteria listed above. Explicitly state where the draft fails or succeeds for each criterion.]

### 3. Revisions Being Made
[Create a bulleted list of the exact, actionable changes you will make based on your critique.]

### 4. Final Answer
[Provide the final, polished response incorporating all of your revisions. This must be the best possible version of the answer.]
"""

def ask_reflecting_bot(question: str):
    print(f"User Question: {question}\n")
    print("Generating response with active self-reflection...\n" + "="*50)
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=reflection_system_instruction,
            temperature=0.4, 
        )
    )
    
    print(response.text.strip())
    print("\n" + "="*50)

# --- TEST THE AGENT ---
# We will ask a question that requires formatting and a specific audience to trigger a good critique
test_question = "Explain the concept of 'Cloud Computing' to a 75-year-old grandparent who only uses a computer to check email. Keep it brief."
ask_reflecting_bot(test_question)