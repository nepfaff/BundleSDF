## Installation script that was written for Ubuntu 22.04 with CUDA 12.1

#!/bin/bash
set -e

# Set timezone
export TZ=US/Pacific

# Set installation prefix
INSTALL_PREFIX=$(pwd)/local
mkdir -p $INSTALL_PREFIX

# Update package lists
sudo apt-get update --fix-missing

# Install required system packages
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    yasm \
    libgtk2.0-dev \
    libgtk-3-dev \
    libjpeg8-dev \
    libtiff5-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-dev \
    libxine2-dev \
    libv4l-dev \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libtbb-dev \
    libatlas-base-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    x264 \
    v4l-utils \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgphoto2-dev \
    libhdf5-dev \
    doxygen \
    libflann-dev \
    libboost-all-dev \
    proj-data \
    libproj-dev \
    libyaml-cpp-dev \
    libzmq3-dev \
    freeglut3-dev \
    rsync \
    lbzip2 \
    pigz \
    zip \
    p7zip-full \
    p7zip-rar \
    software-properties-common

# Set up environment variables for local installation
export PATH=$INSTALL_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export C_INCLUDE_PATH=$INSTALL_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$INSTALL_PREFIX/include:$CPLUS_INCLUDE_PATH

# Create and activate virtualenv named .venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

pip install scikit-image==0.17.2 networkx==2.2

# Install required Python packages
pip install wheel setuptools

# Install required pip packages
pip install torch==2.4.0 torchvision torchaudio

pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl

pip install trimesh opencv-python wandb matplotlib imageio tqdm open3d ruamel.yaml sacred kornia pymongo jupyterlab ninja

pip install pyrender PyOpenGL-accelerate

pip install scipy

# Set compiler variables
export CC=$(which gcc)
export CXX=$(which g++)

# Build and install Eigen
cd $INSTALL_PREFIX
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
make install

# Build and install OpenCV
cd $INSTALL_PREFIX
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout master  # or a specific branch or tag for OpenCV 5 if available

cd $INSTALL_PREFIX
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout master  # or the corresponding branch/tag for OpenCV 5

# Ensure the OpenCV directory exists before proceeding
if [ -d "$INSTALL_PREFIX/opencv" ]; then
    cd $INSTALL_PREFIX/opencv
else
    echo "Error: Directory $INSTALL_PREFIX/opencv does not exist. Cloning may have failed."
    exit 1
fi

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CUDA_STUBS=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_MATLAB=OFF \
    -DCUDA_FAST_MATH=ON \
    -DMKL_WITH_OPENMP=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_OPENMP=ON \
    -DWITH_QT=ON \
    -DWITH_OPENEXR=ON \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    -DBUILD_opencv_cudacodec=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DCMAKE_CXX_FLAGS="-std=c++17 -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT" \
    -DBUILD_opencv_xfeatures2d=ON \
    -DOPENCV_DNN_OPENCL=OFF \
    -DWITH_CUDA=ON \
    -DWITH_OPENCL=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

make -j$(nproc)
make install

# Build and install PCL
cd $INSTALL_PREFIX
git clone https://github.com/PointCloudLibrary/pcl
cd pcl
git checkout pcl-1.10.0

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_apps=OFF \
    -DBUILD_GPU=OFF \
    -DBUILD_CUDA=OFF \
    -DBUILD_examples=OFF \
    -DBUILD_global_tests=OFF \
    -DBUILD_simulation=OFF \
    -DCUDA_BUILD_EMULATION=OFF \
    -DCMAKE_CXX_FLAGS=-std=c++17 \
    -DPCL_ENABLE_SSE=ON \
    -DPCL_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

make -j$(nproc)
make install

# Build and install pybind11
cd $INSTALL_PREFIX
git clone https://github.com/pybind/pybind11
cd pybind11
git checkout v2.10.0

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DPYBIND11_INSTALL=ON \
    -DPYBIND11_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

make -j$(nproc)
make install

# Build and install yaml-cpp
cd $INSTALL_PREFIX
git clone https://github.com/jbeder/yaml-cpp
cd yaml-cpp
git checkout yaml-cpp-0.7.0

mkdir build
cd build

cmake .. -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DINSTALL_GTEST=OFF \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_BUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

make -j$(nproc)
make install

# Clone and build Kaolin
cd $INSTALL_PREFIX
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin

cd kaolin

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OPENCV_IO_ENABLE_OPENEXR=1

# Install freeimage binaries
python -c "import imageio; imageio.plugins.freeimage.download()"

# Build Kaolin
export FORCE_CUDA=1
python setup.py develop

# Install additional pip packages
pip install transformations einops scikit-image awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui pytinyrenderer pyqt5 cython-npm chardet openpyxl

ROOT=$(pwd)
cd ${INSTALL_PREFIX}/kaolin && pip install -e .
cd ${ROOT}/mycuda && rm -rf build *egg* && pip install -e .
cd ${ROOT}/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j11

# Do later to prevent errors from dependencies.
pip install --upgrade networkx

# All done
echo "Installation complete."