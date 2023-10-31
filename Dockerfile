FROM python

# set work directory
WORKDIR /root/real_estate_prices

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
COPY ./requirements.txt /root/real_estate_prices/requirements.txt
RUN pip install -r requirements.txt