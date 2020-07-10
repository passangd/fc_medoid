FROM python:3

RUN apt-get update \
    && apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev

WORKDIR /usr/src/fc
COPY . . 
EXPOSE 80
RUN pip install -r requirements.txt
RUN python src/compile_medoids.py
ENTRYPOINT ["python", "src/main.py"]
CMD ["--help"] 
