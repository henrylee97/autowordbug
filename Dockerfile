FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY autowordbug /src/autowordbug
COPY *.py /src/

RUN sed -i '1s/^/#!\/opt\/conda\/bin\/python\n/' /src/train.py
RUN sed -i '1s/^/#!\/opt\/conda\/bin\/python\n/' /src/prepare.py
RUN sed -i '1s/^/#!\/opt\/conda\/bin\/python\n/' /src/evaluate.py

RUN chmod +x /src/*.py

RUN ln -s /src/train.py /usr/local/bin/train
RUN ln -s /src/prepare.py /usr/local/bin/prepare
RUN ln -s /src/evaluate.py /usr/local/bin/evaluate

CMD [ "bash" ]