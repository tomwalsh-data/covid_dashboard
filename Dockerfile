# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04
# WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8050
COPY . .
# CMD ["ls", "code"]
#CMD [ "python3", "-m", "code/covid_dashboard.py", "run", "--host=0.0.0.0"]
CMD ["python3", "covid_dashboard.py"]
# ${WORKDIR}/covid_dashboard.py"]