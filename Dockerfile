FROM python:3.8-slim-buster

# Install the security updates.
RUN apt-get update
RUN apt-get -y upgrade

RUN pip install -y --upgrade pip setuptools wheel
RUN apt-get -y libxml2-dev libxmlsec1-dev

# Remove all cached file. Get a smaller image.
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

EXPOSE 8501

# Copy the application.
COPY . /opt/app
WORKDIR /opt/app

RUN mkdir ~/.streamlit

# Install the app librairies.
RUN pip install -r requirements.txt

# Start the app.
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "app.py" ]