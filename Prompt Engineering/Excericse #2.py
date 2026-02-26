import os
import re
from google import genai
from google.genai import types

# Hardcode your API key for the assignment
my_api_key = "API_KEY_HERE"
client = genai.Client(api_key=my_api_key)

MODEL_ID = "gemini-2.5-flash"

# --- 1. DEFINE OUR AGENT'S TOOLS ---
# These are the Python functions the AI can choose to "run" during its Action phase.

def get_rent_estimates(housing_type: str) -> str:
    """Returns average monthly costs for different student housing options."""
    costs = {
        "dorm": "San Jose State University Dorms: ~$1,800/month (includes utilities, wifi, and basic meal plan).",
        "apartment": "Off-Campus Apartment (San Jose area): ~$1,500 - $2,200/month for a private room (plus ~$150 for utilities and groceries).",
        "home": "Living at Home: $0 - $400/month (mostly gas/transit costs for commuting)."
    }
    # Default to a general summary if they don't specify exactly
    if housing_type.lower() not in costs:
        return f"Available categories are: dorm, apartment, home. Please specify one."
    return costs[housing_type.lower()]

def get_commute_impact(distance_miles: str) -> str:
    """Returns a general statement on how commute time impacts student life."""
    miles = int(distance_miles)
    if miles <= 2:
        return "Excellent. Walking/biking distance. Easy to join clubs, study groups, and access the library late at night."
    elif miles <= 15:
        return "Moderate. 20-40 minute drive or transit. Requires planning classes in blocks to avoid multiple trips. Might miss some spontaneous campus events."
    else:
        return "Heavy. 45+ minute commute. High risk of burnout. Difficult to participate in extracurriculars. Requires a reliable car or strict transit schedule."

# A dictionary to map the AI's string output to our actual Python functions
AVAILABLE_TOOLS = {
    "get_rent_estimates": get_rent_estimates,
    "get_commute_impact": get_commute_impact
}

# --- 2. THE REACT SYSTEM PROMPT ---
# This strict formatting forces the LLM to think, act, and observe.

react_system_instruction = """
You are a college housing advisor AI. You help students decide between living in a Dorm, an Off-Campus Apartment, or at Home.

You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you output an Answer.

Use Thought to describe your reasoning about the student's situation.
Use Action to run one of the tools available to you. Return EXACTLY in this format:
Action: tool_name: input_variable

After an Action, you must output PAUSE. Do not generate the Observation yourself. 

Available tools:
- get_rent_estimates: Pass 'dorm', 'apartment', or 'home' to get cost data.
- get_commute_impact: Pass the estimated miles from campus (e.g., '10') to see how it affects student life.

Example Session:
Question: I live 12 miles away and have a tight budget. Should I get an apartment or live at home?
Thought: The student is on a tight budget. I need to check the cost of an apartment vs living at home.
Action: get_rent_estimates: apartment
PAUSE

Observation: Off-Campus Apartment: ~$1,500 - $2,200/month.
Thought: Now I need to check the cost of living at home.
Action: get_rent_estimates: home
PAUSE

Observation: Living at Home: $0 - $400/month.
Thought: Living at home is much cheaper, but they are 12 miles away. Let me check the commute impact.
Action: get_commute_impact: 12
PAUSE

Observation: Moderate. 20-40 minute drive. Requires planning.
Final Answer: Since you have a strict budget, living at home is your safest financial bet ($0-$400 vs $1500+). However, because you are 12 miles away, be prepared to schedule your classes in blocks so you aren't driving back and forth multiple times a day!
"""

# --- 3. THE REACT EXECUTION LOOP ---

def run_react_agent(user_query: str):
    print(f"\n[User Query]: {user_query}")
    print("-" * 50)
    
    # We maintain the prompt dynamically as the agent "thinks" and "acts"
    current_prompt = f"Question: {user_query}\n"
    
    # Cap the loop at 5 iterations so it doesn't get stuck in an infinite loop
    for i in range(5):
        # 1. Ask the LLM what to do next
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=current_prompt,
            config=types.GenerateContentConfig(
                system_instruction=react_system_instruction,
                temperature=0.2, # Keep it low so it follows instructions strictly
                stop_sequences=["Observation:"] # Force it to stop before hallucinating tool results!
            )
        )
        
        agent_output = response.text.strip()
        print(f"\n{agent_output}")
        current_prompt += f"\n{agent_output}\n"

        # 2. Check if the agent wants to give the Final Answer
        if "Final Answer:" in agent_output:
            print("\n✅ Task Complete.")
            return

        # 3. Check if the agent wants to take an Action
        action_match = re.search(r"Action:\s*([a-zA-Z_]+):\s*(.+)", agent_output)
        
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = action_match.group(2).strip()
            
            print(f"\n[System executing tool: {tool_name}({tool_input})]")
            
            # 4. Execute the Python tool
            if tool_name in AVAILABLE_TOOLS:
                tool_result = AVAILABLE_TOOLS[tool_name](tool_input)
            else:
                tool_result = f"Error: Tool '{tool_name}' not found."
                
            # 5. Feed the Observation back into the prompt for the next loop
            observation_text = f"Observation: {tool_result}"
            print(f"[{observation_text}]")
            current_prompt += f"{observation_text}\n"
        else:
            print("\n❌ Error: Agent failed to format an Action or Final Answer correctly.")
            break

# --- TEST THE AGENT ---
test_question = "I have a part-time job that pays well, but I really want to be involved in campus clubs. I currently live 20 miles away from campus. What should I do?"
run_react_agent(test_question)

