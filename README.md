
# go here...
cd workspace/
cd ai_research_assistant/

# create a virtual environment
python --version
python -m venv venv

LINUX
source venv/bin/activate
WINDOOF
$ source venv/Scripts/activate
(venv)
C144657@C070093 MINGW64 ~/workspace/ai_research_assistant


# Install the required packages
pip install langchain openai beatuifulsoup4 requests faiss-cpu langchain-community tiktoken langchain-openai beautifulsoup4
Collecting langchain
Downloading langchain-0.3.26-py3-none-any.whl.metadata (7.8 kB)
Collecting openai
Downloading openai-1.93.2-py3-none-any.whl.metadata (29 kB)
...


# run the script
python ai_research_assistant.py