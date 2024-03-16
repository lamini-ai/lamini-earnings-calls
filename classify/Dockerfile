# Base container name
ARG BASE_NAME=python:3.11

FROM $BASE_NAME as base

ARG PACKAGE_NAME="lamini-earnings-classify"

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt

RUN pip install -r requirements.txt

# Copy all files to the container
COPY scripts /app/${PACKAGE_NAME}/scripts
COPY lamini_earnings_classify /app/${PACKAGE_NAME}/lamini_earnings_classify
WORKDIR /app/${PACKAGE_NAME}

ENV PACKAGE_NAME=$PACKAGE_NAME
ENTRYPOINT ["/app/lamini-earnings-classify/scripts/start.sh"]


