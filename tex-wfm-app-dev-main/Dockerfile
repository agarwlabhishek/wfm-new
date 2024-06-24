FROM python:3.8-slim-bookworm

# set proxy variables
ARG http_proxy_arg
ARG no_proxy_arg
ENV no_proxy $no_proxy_arg
ENV http_proxy $http_proxy_arg
ENV https_proxy $http_proxy_arg

# install requirements and configure the app
ADD app/ /app/
COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

WORKDIR /app
EXPOSE 8050
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "main.py" ]
