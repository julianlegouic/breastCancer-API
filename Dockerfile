FROM tiangolo/uvicorn-gunicorn:python3.6-alpine3.8

# Make directories suited to your application
RUN mkdir -p /home/project/app
WORKDIR /home/project/app

# Copy and install requirements
COPY requirements.txt /home/project
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN pip install --no-cache-dir -r ../requirements.txt

# Copy contents from your local to your docker container
COPY ./app /home/project/app