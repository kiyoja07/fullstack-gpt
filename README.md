OpenAI API
https://openai.com/blog/openai-api

#2.5 Virtual Environment
가상 환경 세팅

- requirement.txt 최초 버전 : https://gist.github.com/serranoarevalo/72d77c36dde1cc3ffec34105eb666140

- 기존 가상 환경 삭제 : rm -rf env
- 설치 : python3.11 -m venv ./env
- 활성화 : source env/bin/activate
- 패키지 설치 : pip install -r requirement.txt
- 비활성화 : deactivate

#6.5 Langsmith

- https://smith.langchain.com

#7.0 Introduction

- 가상 환경 활성화 : source env/bin/activate
- streamlit 파일 실행 : streamlit run HOME.py

source env/bin/activate && streamlit run HOME.py

#8.4 Ollama

- langchain 패키지 설치 : pip install langchain-ollama
- ollama 다운로드 : https://ollama.com
- 터미널로 deepseek-r1:1.5b 모델 (로컬에 없으면) 설치 및 실행 : ollama run deepseek-r1:1.5b

#10.1 AsyncChromiumLoader

- source env/bin/activate
- pip install pytest-playwright
- playwright install

#11.1 Audio Extraction

- brew install ffmpeg
- ffmpeg -i files/podcast.mp4 -vn files/audio.mp3
