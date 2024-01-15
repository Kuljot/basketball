# Basketball Detector
## Basketball score detection using YOLOV8 X DeepSort

[![N|YOLOV8](https://assets-global.website-files.com/6479eab6eb2ed5e597810e9e/65342ba375a1412e2f259dcc_LinkedIn_Product_Page_YOLOv8-p-800.png)](https://assets-global.website-files.com/6479eab6eb2ed5e597810e9e/65342ba375a1412e2f259dcc_LinkedIn_Product_Page_YOLOv8-p-800.png)

Basketball score lets you count the score and detect Players, Ball and the Hoop(Rim) from any video,
The container can be deployed on any PC in the network and used from the streamlit web-app for real time detection.

- The app uses streamlit as frontend
- Tensorflow, Ultralytics Yolo V8, DeepSort in the backend

## Features

- Convert any basketball video to detect the score and detect players, ball and hoop(Rim)
- Use the web app over the network for real time detection, 

Any suggestions are welocme you can contact me 
 [Kuljot Singh] on my email [kuljotme035@gmail.com]

> There is one known issue when the
> container is deployed on cloud e.g. AWS the
> video stream cant be enabled irrespective of 
> virtual or bare metal instance. 
> Currently only local networks are supported


## Tech

The program uses a number of open source libraries to work :

- [Streamlit](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiKzMaevJuDAxXITWwGHQCFArsQFnoECAYQAQ&url=https%3A%2F%2Fstreamlit.io%2F&usg=AOvVaw0COPYHEMKG9SPXbyFDXyMf&opi=89978449) -  For frontend of the web apps!
- [SciKit Learn](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjY5N2RvJuDAxWybmwGHa7uAksQFnoECAcQAQ&url=https%3A%2F%2Fscikit-learn.org%2F&usg=AOvVaw3pidYsGhglQXGDh_4GMetL&opi=89978449) - Make and train the ML models
- [Numpy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjw6p75u5uDAxXRTWwGHch4AbQQFnoECAoQAQ&url=https%3A%2F%2Fnumpy.org%2F&usg=AOvVaw3L2i9HVc9ZeynETpNrPxO-&opi=89978449) - for numerical processing
- [Pandas](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiVzuHpu5uDAxU5TWwGHRszBdkQFnoECAUQAQ&url=https%3A%2F%2Fpandas.pydata.org%2F&usg=AOvVaw3cD5ulu4AnZcNusojIyttY&opi=89978449) - handeling the dataframes
- [DeepSort](https://github.com/nwojke/deep_sort) - For keeping track of the detected objects
- [ssl-proxy](https://github.com/suyashkumar/ssl-proxy) - For creating ssl proxy
- [YOLO V8](https://github.com/ultralytics/ultralytics) - A FCNN based model for detecting objects in the images
- [OpenCV](https://opencv.org/) - For drawing rectangles on the video


And it is open source project with a [https://github.com/Kuljot/basketball] on GitHub. Built as an educational project. Please treat it for educational purpose only, PLEASE DONT UPLOAD ANY SENSITIVE/PERSONAL INFO.

## Installation on your local Machine

CSVtoML requires Python 3.11+ to run.

Clone the repository on your device
```sh
git clone https://github.com/Kuljot/basketball
cd basketball
```
Create a virtual environment and activate it
```sh
sudo apt install python3.11-venv
python3.11 -m venv env
source env/bin/activate
```

Install the requirements
```sh
pip install -r requirements.txt
```
Step 1:
Enable ssl proxy, it makes the port connection support https, else browser wont allow video stream 
If your system is x86 type then use:
```sh
./ssl-proxy-linux-amd64  -from 0.0.0.0:443 -to 127.0.0.1:8501 &
```
else if your system is arm type use:
```sh
apt update
apt install -y qemu-user-static binfmt-support 
dpkg --add-architecture amd64 
apt update 
apt install -y libc6:amd64 
qemu-x86_64-static src/ssl-proxy-linux-amd64 -from 0.0.0.0:443 -to 127.0.0.1:8501 &
```

Step:2 
Run the application
```sh
streamlit run app.py
```

## Using Docker

The program can be containerized via docker.

By default, the Docker will expose port 8080 and streamlit uses 8051, 
but I have exposed the port 443 explicitly only because its default 
exposed port on AWS, any other port can also be used. 

Simply use the Dockerfile to build the image for x86 type processor based system.

```sh
sudo docker build -t basketball .
```
Else if you have a arm type processor
Change the Docker file by adding before the CMD command 

```
RUN apt update
RUN apt install -y qemu-user-static binfmt-support 
RUN dpkg --add-architecture amd64 
RUN apt update 
RUN apt install -y libc6:amd64 
```

Change the entrypoint.sh before building image.

```sh
#!/bin/bash

echo "Starting application..."
qemu-x86_64-static src/ssl-proxy-linux-amd64 -from 0.0.0.0:443 -to 127.0.0.1:8501 &
streamlit run app.py
```

then build the image
```sh
sudo docker build -t basketball .
```

Verify the app by navigating to your server address in
your preferred browser.

```sh
http://localhost:443/
```

For using it as a video file converter only use command
```sh
python convert.py  path_to_video_file  path_to_the_yolo_model_file
```

## Dataset Used:
https://universe.roboflow.com/ownprojects/basketball-w2xcw/dataset/2/images
Model file is not shared due to size.
## License
CC
**OpenSource!**

