# start from python base image
FROM python:3.11

# change working directory
WORKDIR /code

# add requirements file to image
COPY . .

# install python libraries
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# specify default commands
CMD ["fastapi", "run", "main.py", "--port", "80"]