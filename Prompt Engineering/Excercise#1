import os
from google import genai
from google.genai import types

# Hardcode your API key for the assignment
my_api_key = "API_KEY_HERE"
client = genai.Client(api_key=my_api_key)

MODEL_ID = "gemini-2.5-flash"

# --- MOLLY TEA KNOWLEDGE BASE ---
MOLLY_TEA_MENU = """
Molly Tea (茉莉奶白) Specialties:
- Jasmine Snow Bud: Signature jasmine green tea with fresh milk and a snow cream top.
- White Peach Oolong: Refreshing oolong tea infused with white peach flavor.
- Gardenia Green Tea: Floral green tea with a delicate gardenia scent, available with or without milk.
- Matcha Snow: Premium matcha blended with milk and topped with signature cream.
Customizations: Ice levels (regular, less, no ice), Sugar levels (regular, half, slight, zero).
"""

# --- STEP 1: CLASSIFY ISSUE ---
def step_1_classify_intent(chat_history_text: str) -> str:
    """Classifies the overall conversation intent."""
    system_instruction = "You are a routing assistant. Look at the conversation history and classify the MAIN goal of the customer into ONE category: 'Order_Issue', 'Menu_Question', or 'General_Inquiry'. Output ONLY the exact category name."
    
    prompt = f"Conversation History:\n{chat_history_text}\n\nWhat is the main intent of this conversation?"
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0)
    )
    return response.text.strip()

# --- STEP 2: GATHER MISSING INFO ---
def step_2_extract_info(chat_history_text: str, intent: str) -> str:
    """Extracts required information from the entire chat history."""
    if intent != "Order_Issue":
        return "Not Applicable"
        
    system_instruction = """
    You are an extraction assistant. Check the ENTIRE conversation history to see if the customer has provided an Order Number at ANY point.
    If they did, output ONLY the order number. 
    If they did not, output EXACTLY: 'MISSING_ORDER_NUMBER'.
    """
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=f"Conversation History:\n{chat_history_text}",
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0)
    )
    return response.text.strip()

# --- STEP 3: PROPOSE SOLUTION ---
def step_3_generate_response(chat_history_text: str, intent: str, extracted_info: str) -> str:
    """Generates the final response using context and history."""
    system_instruction = """
    You are a customer support agent for Molly Tea. 
    Your tone must be polite, empathetic, and action-oriented. 
    - DO NOT repeat the customer's problem back to them. 
    - Be concise and professional.
    - Look at the conversation history to ensure you don't repeat questions you've already asked.
    """
    
    if intent == "Menu_Question":
        context = f"Use this menu knowledge to assist the customer:\n{MOLLY_TEA_MENU}"
    elif intent == "Order_Issue":
        if extracted_info == "MISSING_ORDER_NUMBER":
            context = "The customer has an order issue but hasn't provided an order number yet. Ask: 'We're sorry to hear that! Could you please provide your order number so we can investigate this for you?'"
        else:
            context = f"We have the order number: {extracted_info}. If the customer hasn't provided a photo yet, apologize and ask for one. If they HAVE provided a photo (or said they are uploading one), thank them and say you will process the replacement."
    else:
        context = "Assist the customer with their general inquiry politely."

    prompt = f"Context Guidelines: {context}\n\nConversation History:\n{chat_history_text}\n\nGenerate the Agent's NEXT logical response ONLY:"
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7)
    )
    return response.text.strip()

# --- INTERACTIVE CHAT LOOP ---
print("Welcome to the Molly Tea Support Test Environment!")
print("Type 'exit' or 'quit' to stop.\n")


# BEFORE: Chat bot did not save memory from previous messages
# AFTER: Chat Bot will now save memory from previous message so it can provide a solution

# This list will act as our bot's memory!
chat_history = []

while True:
    user_input = input("You (Customer): ")
    
    if user_input.lower() in ['exit', 'quit', '']:
        print("Ending test session. See ya!")
        break
    
    # Add the user's newest message to the memory
    chat_history.append(f"Customer: {user_input}")
    
    # Convert the memory list into a single block of text so the AI can read it
    history_text = "\n".join(chat_history)
    
    print(f"\n[System] Running Chain with Memory...")
    
    # Pass the ENTIRE history into the chain, not just the single message
    intent = step_1_classify_intent(history_text)
    print(f"  -> [Step 1] Intent: {intent}")
    
    extracted_info = step_2_extract_info(history_text, intent)
    print(f"  -> [Step 2] Extracted Info: {extracted_info}")
    
    response = step_3_generate_response(history_text, intent, extracted_info)
    print(f"\nAgent: {response}")
    
    # Add the agent's response to the memory so it remembers what it said!
    chat_history.append(f"Agent: {response}")
    print("-" * 50)
