import sys
print("Python version:", sys.version)
print("Testing imports step by step...")

try:
    print("1. Testing basic imports...")
    import json
    from typing import Any, Dict, List, Optional, Union
    print("   [OK] Basic imports successful")
    
    print("2. Testing LangChain core...")
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    print("   [OK] LangChain core successful")
    
    print("3. Testing LangChain community...")
    from langchain_community.chat_models import ChatOpenAI
    print("   [OK] LangChain community successful")
    
    print("4. Testing LangChain tools...")
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    print("   [OK] LangChain tools successful")
    
    print("5. Testing API settings...")
    from api_setting import API_KEY, API_URL, API_MODEL
    print("   [OK] API settings successful, Model:", API_MODEL)
    
    print("6. Testing enhanced agents...")
    from enhanced_agents import EnhancedAgent, run_enhanced_agent
    print("   [OK] Enhanced agents successful")
    
    print("\n[SUCCESS] All imports completed successfully!")
    
    print("7. Testing agent creation...")
    agent = EnhancedAgent()
    print("   [OK] Agent instance created")
    
    print("\n[READY] Enhanced agent is ready to use!")
    
except Exception as e:
    print("\n[ERROR] Import failed:", str(e))
    import traceback
    traceback.print_exc()
