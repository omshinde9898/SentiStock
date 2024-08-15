FROM python:3.12.5-bullseye
COPY . .
CMD ["pip","install","-r","requirements.txt"]
RUN ["streamlit","app.py"]