FROM nvcr.io/nvidia/pytorch:21.12-py3

WORKDIR /workspace

# Install python requirements
RUN pip install --upgrade pip
# RUN pip install torch torchvision torchaudio
RUN pip3 install fairseq==v0.9.0

RUN pip install OmegaConf==1.4.1
RUN pip install hydra-core==0.11
RUN pip install tensorboardX
RUN pip install tensorboard
RUN pip install soundfile
RUN pip install tqdm
RUN pip install scipy
RUN pip install matplotlib
RUN pip install scikit-learn==0.22.0
RUN pip install frozendict
RUN pip install pandas
RUN pip install submitit
RUN pip install rtpt

SHELL ["/bin/bash", "-c"]

# echo current path
RUN echo $PWD

# set environment variable
# RUN ${RUN_DIR}=runs

COPY paths.sh .
# run source paths.sh
RUN source paths.sh

