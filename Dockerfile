FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-trt8.2.5

WORKDIR /opt/ml/wxcode

COPY ./requirements.txt ./
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

COPY ./save/model.trt.engine ./model.trt.engine
COPY ./opensource_models/chinese-roberta-wwm-ext/ ./opensource_models/chinese-roberta-wwm-ext/
COPY ./utils/*.py ./utils/
COPY ./*.py ./
COPY ./start.sh ./

CMD sh -c "sh start.sh"
