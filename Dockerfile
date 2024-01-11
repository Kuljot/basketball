# Dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y

EXPOSE 443


# Set the entrypoint script as executable
RUN chmod +x /app/entrypoint.sh

# Define the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]




