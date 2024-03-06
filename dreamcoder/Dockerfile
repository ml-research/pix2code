FROM nvcr.io/nvidia/pytorch:21.07-py3 

SHELL ["/bin/bash", "-c"]

# Install packages 
RUN apt-get update && apt-get -y upgrade && apt-get -y install git \
    wget \
    m4 \
    libcairo2-dev \
    libzmq3-dev \
    swig \
    graphviz  \
    opam 

# Opam  setup
RUN opam init --disable-sandboxing
RUN eval $(opam env)
RUN opam update
RUN eval $(opam env)
RUN opam switch
RUN eval $(opam env)
RUN opam switch create 4.06.1+flambda
RUN eval $(opam env)
RUN eval 'opam config env'
# need fixed older versions for yojson and mehir to be able to build dreamcoders ocaml libraries (see repos issues)
RUN opam install -y ppx_jane core re2 yojson.1.6.0 vg cairo2 camlimages menhir.20211128 ocaml-protoc zmq utop jbuilder

WORKDIR /workspace

COPY requirements.txt .

# Install python requirements
RUN pip install -r requirements.txt

# Install rtpt
RUN pip install rtpt
