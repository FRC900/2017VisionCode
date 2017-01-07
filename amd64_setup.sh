sudo apt-get update
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-375
cd
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
chmod 755 cuda_8.0.44_linux-run
sudo ./cuda_8.0.44_linux-run --silent --toolkit --samples --samplespath=/usr/local/cuda-8.0/NVIDIA_CUDA-8.0_Samples
rm cuda_8.0.44_linux-run
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# This seems to fail - cuDNN download requires authentication?
wget http://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/8.0/cudnn-8.0-linux-x64-v5.1-tgz
cd /usr/local
sudo tar -xzvf ~/cudnn-8.0-linux-x64-v5.1-tgz
rm ~/cudnn-8.0-linux-x64-v5.1-tgz

cd
wget https://www.stereolabs.com/download_327af3/ZED_SDK_Linux_Ubuntu16_CUDA80_v1.2.0.run
chmod 755 ZED_SDK_Linux_Ubuntu16_CUDA80_v1.2.0.run
./ZED_SDK_Linux_Ubuntu16_CUDA80_v1.2.0.run

# Install ffmpeg. This is a prereq for OpenCV, so 
# unless you're installing that skip this as well.
sudo apt-get install yasm
cd
wget --no-check-certificate https://github.com/FFmpeg/FFmpeg/archive/n3.1.3.zip
unzip n3.1.3.zip
cd FFmpeg-n3.1.3
./configure --enable-shared
make -j4
sudo make install
cd ..
rm -rf FFmpeg-n3.1.3 n3.1.3.zip

# OpenCV build info. Not needed for Jetson, might be
# needed for x86 PCs to enable CUDA support 
# Note that the latest ZED drivers for x86_64 require
# OpenCV3.1 install should be similar, just download the
# correct version of the code
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
cd
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -DCUDA_ARCH_BIN="5.2 6.1" -DCUDA_ARCH_PTX="5.2 6.1" -DOPENCV_EXTRA_MODULES_PATH=/home/ubuntu/opencv_contrib/modules -DBUILD_opencv_dnn=OFF ..
make -j4
sudo make install

# Install nv caffe fork - needed for DIGITS
sudo apt-get install build-essential cmake git gfortran libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev protobuf-compiler python-all-dev python-dev python-h5py python-matplotlib python-numpy python-pil python-pip python-protobuf python-scipy python-skimage python-sklearn
pip install --upgrade pip
cd
git clone https://github.com/NVIDIA/caffe.git caffe-nv
cd caffe-nv
cat python/requirements.txt | xargs -n1 sudo pip install --upgrade
mkdir build
cd build
cmake -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="52 61" -DCUDA_ARCH_PTX="52 61" ..
make -j6 install

# Install DIGITS itself
cd
git clone https://github.com/NVIDIA/DIGITS.git
cd DIGITS
sudo apt-get install git graphviz python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-pip python-protobuf python-scipy
sudo pip install --upgrade -r requirements.txt
sudo pip install --upgrade -r requirements_test.txt

echo "export CAFFE_ROOT=/home/ubuntu/caffe-nv" >> ~/.bashrc
export CAFFE_ROOT=/home/ubuntu/caffe-nv
