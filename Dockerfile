# Dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y

EXPOSE 8000
#CMD ["src/ssl-proxy-linux-amd64","-from","0.0.0.0:8000","-to","127.0.0.1:8501","&","&&","streamlit", "run", "app.py"]
#RUN echo "#!/bin/bash\n\
#          src/ssl-proxy-linux-amd64 -from 0.0.0.0:8000 -to 127.0.0.1:8501 &\n\
#          streamlit run app.py" > run.sh

# Make the script executable
#RUN chmod +x run.sh

CMD ["src/ssl-proxy-linux-amd64", "-from", "0.0.0.0:8000", "-to", "127.0.0.1:8501","&"] 

# Define the command to run the script
CMD ["/bin/bash", "run.sh"]


#
