#set python environment

FROM python:3

#set Google Cloud credentials env variabl to IAM account key
ENV GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

#install required packages and libraries from requirements.txt
RUN echo "Installing required packages and libraries"
RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]

#RUN python3 main.py


#$ docker build -t my-python-app .
#$ docker run -it --rm --name my-running-app my-python-app
