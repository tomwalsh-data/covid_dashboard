# syntax=docker/dockerfile:1
# FROM python:3.7-alpine
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
WORKDIR /code
# ENV MY_APP=covid_dashboard.py
# ENV APP_RUN_HOST=0.0.0.0
# RUN apk add --no-cache gcc musl-dev linux-headers 
#geos gdal 
# gdal
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["python3", "./code/covid_dashboard.py"]