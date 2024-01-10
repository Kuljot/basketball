# Dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
#CMD ["src/ssl-proxy-linux-amd64","-from","0.0.0.0:8000","-to","127.0.0.1:8501","&","&&","streamlit", "run", "app.py"]
RUN echo "#!/bin/bash\n\
          src/ssl-proxy-linux-amd64 -from 0.0.0.0:8000 -to 127.0.0.1:8501 &\n\
          streamlit run app.py" > run.sh

# Make the script executable
RUN chmod +x run.sh

# Define the command to run the script
CMD ["/bin/bash", "run.sh"]