FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    openssh-server \
    sudo \
    && apt-get clean

# Binding Python do MPI (pacote do Ubuntu, ja compilado contra o OpenMPI do no)
RUN apt-get update && apt-get install -y python3-mpi4py && apt-get clean

# Configuração do SSH
RUN mkdir /var/run/sshd
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Criação do usuário exigido pelo laboratório
RUN useradd -m -s /bin/bash mpiuser
RUN echo 'mpiuser:mpi' | chpasswd

# Geração de chave SSH na imagem (todos os contêineres compartilharão a mesma chave)
USER mpiuser
RUN ssh-keygen -t rsa -f /home/mpiuser/.ssh/id_rsa -q -N ""
RUN cp /home/mpiuser/.ssh/id_rsa.pub /home/mpiuser/.ssh/authorized_keys

USER root
WORKDIR /home/mpiuser
CMD ["tail", "-f", "/dev/null"]
