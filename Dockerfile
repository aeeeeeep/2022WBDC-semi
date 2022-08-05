FROM ccr.ccs.tencentyun.com/ti_containers/pytorch:1.9.1-gpu-cu111-py38

WORKDIR /opt/ml/wxcode

COPY ./ ./
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple