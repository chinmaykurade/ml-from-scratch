FROM python:3.7.11

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/ml_api

ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./packages/ml_api /opt/ml_api/
# To avoid cv2 import errors
RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1
RUN apt-get install ffmpeg -y

RUN pip install --upgrade pip
RUN pip install -r /opt/ml_api/requirements.txt

RUN python /opt/ml_api/manage.py makemigrations
RUN python /opt/ml_api/manage.py migrate

RUN chmod +x /opt/ml_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 5000

CMD ["bash", "./run.sh"]

