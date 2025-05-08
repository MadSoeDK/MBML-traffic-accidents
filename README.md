# Model-based Machine Learning

To build a docker image with a python environment with necessary dependencies
```
docker build -t mbml-project:latest .
```

To run the container
```
docker run -d \
  -p 8888:8888 \
  -v "$(pwd)":/home/jovyan/ \
  --name mbml-project \
  mbml-project:latest
```