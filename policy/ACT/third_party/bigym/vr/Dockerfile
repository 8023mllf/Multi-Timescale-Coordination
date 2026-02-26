FROM nvidia/vulkan:1.3-470

ARG UID
ARG GID
ARG USER

RUN dpkg --add-architecture i386
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes apt-utils
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes mesa-utils xauth kmod \
     wget gdebi-core pciutils usbutils dbus-x11 psmisc strace libdbus-1-3:i386 \
     libdbus-glib-1-2:i386 libgl1-mesa-dri:i386 \
     libgl1-mesa-glx:i386 sudo pulseaudio-utils gdebi-core \
     libvulkan1 libvulkan1:i386
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes libnm-dev:i386
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes libgl1:i386 libcap2-bin

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes \
        build-essential \
        libeigen3-dev \
        pkg-config \
        libx11-dev \
        libx11-xcb-dev \
        libxxf86vm-dev \
        libxrandr-dev \
        libxcb-randr0-dev \
        libvulkan-dev \
        mesa-vulkan-drivers:i386 \
        glslang-tools \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        ca-certificates \
        libusb-1.0-0-dev \
        libudev-dev \
        libhidapi-dev \
        libwayland-dev \
        libuvc-dev \
        libavcodec-dev \
        libopencv-dev \
        libv4l-dev \
        libcjson-dev \
        libsdl2-dev \
        libbsd-dev

RUN apt-get update && \
    apt-get install -y git cmake meson ninja-build g++ libgl1-mesa-dev libjsoncpp-dev && \
    git clone https://github.com/KhronosGroup/OpenXR-SDK.git && \
    cd OpenXR-SDK && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr -Bbuild

# Install pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy pulse client config for sound
COPY vr/pulse-client.conf /etc/pulse/client.conf

RUN adduser --shell /bin/bash --system --group --uid ${UID} ${USER} && \
    passwd -d ${USER} && \
    adduser ${USER} sudo

USER ${USER}

# Install Steam
RUN sudo apt-get install -y --no-install-recommends \
  steam-installer && \
  sudo ln -sf /usr/games/steam /usr/bin/steam

# Activate bigym venv and install bigym
RUN conda init && conda create -y -n bigym python=3.10

# Add an alias to run vrmonitor.sh
RUN echo "alias steamvr='$HOME/.steam/debian-installation/steamapps/common/SteamVR/bin/vrmonitor.sh'" >> ~/.bashrc

CMD ["/usr/bin/bash"]
