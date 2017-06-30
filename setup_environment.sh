jetson=true
version=tx
gpu=true

#process args
while [ $# -gt 0 ]
do
	case "$1" in
		-jx) jetson=true;;
	-jk) jetson=true; version=tk;;
	-amd64) jetson=false;;
	-c) gpu=false;;
	-h) echo >&2 \
		"usage: $0 [-jx or -jk or -amd64] [-c] [-h]"
		exit 1;;
	*)  break;;	# terminate while loop
	esac
	shift
done

#install basic dependencies

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y libeigen3-dev build-essential gfortran git cmake libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev liblmdb-dev vim-gtk libgflags-dev libgoogle-glog-dev libatlas-base-dev python-dev python-pip libtinyxml2-dev v4l-conf v4l-utils libgtk2.0-dev pkg-config exfat-fuse exfat-utils libprotobuf-dev protobuf-compiler unzip python-numpy python-scipy python-opencv python-matplotlib chromium-browser wget unzip

sudo apt-get install --no-install-recommends -y libboost-all-dev

#install caffe
cd
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build

if [ "$gpu" == "false" ] ; then
	cmake -DCPU_ONLY=ON ..
else
	cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
fi

make -j4 all
#make test
#make runtest
make -j4 install

# Install libsodium - this is a prereq for zeromq
cd
wget --no-check-certificate https://download.libsodium.org/libsodium/releases/libsodium-1.0.11.tar.gz
tar -zxvf libsodium-1.0.11.tar.gz
cd libsodium-1.0.11
./configure
make -j4 
sudo make install
cd ..
rm -rf libsodium-1.0.11*

# install zeromq
cd
wget --no-check-certificate https://github.com/zeromq/zeromq4-1/releases/download/v4.1.5/zeromq-4.1.5.tar.gz
tar -xzvf zeromq-4.1.5.tar.gz
cd zeromq-4.1.5
./configure
make -j4
sudo make install
cd ..
rm -rf zeromq-4.1.5*
cd /usr/local/include/
sudo wget --no-check-certificate https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp

# Install tinyxml2
cd
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
mkdir build
cd build
cmake ..
make -j4
sudo make install
cd ../..
rm -rf tinyxml2


# Install Point Cloud Library
sudo apt-get install libflann-dev libpcl-dev
#cd
#wget --no-check-certificate https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.0.zip
#unzip pcl-1.8.0.zip
#cd pcl-pcl-1.8.0
#mkdir build
#cd build
#cmake ..
#make -j4
#sudo make install
#cd ../..
#rm -rf pcl-1.8.0.zip pcl-pcl-1.8.0

#install zed sdk
if [ "$version" = tk1 ] && [ "$jetson" = true ] ; then
	ext="ZED_SDK_Linux_JTK1_v1.2.0.run"
elif [ "$version" = tx1 ] && [ "$jetson" = true ] ; then
	ext="ZED_SDK_Linux_JTX1_v1.2.0_64b_JetPack23.run"
else
	ext="ZED_SDK_Linux_Ubuntu16_CUDA80_v1.2.0.run"
fi
wget --no-check-certificate https://www.stereolabs.com/download_327af3/$ext
chmod 755 $ext
./$ext
rm ./$ext

#clone repo
#TODO : rethink this - how are we getting the script if the
#       repo isn't there in the first place?
cd
git clone https://github.com/FRC900/2017VisionCode.git
cd 2017VisionCode
git submodule init
git submodule update

#build stuff
cd libfovis
mkdir build
cd build
cmake ..
make -j4
cd ../..
cd zebravision
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF .
make -j4

#mount and setup autostart script
if [ "$jetson" = true ] ; then
	sudo mkdir /mnt/900_2
	sudo mkdir -p /usr/local/zed/settings
	sudo chmod 755 /usr/local/zed/settings
	sudo cp ~/2017VisionCode/calibration_files/*.conf /usr/local/zed/settings
	sudo chmod 644 /usr/local/zed/settings/*

	sudo mkdir -p /lib/modules/4.4.15-tegra/kernel/drivers/usb/serial
	sudo cp cp210x.ko /lib/modules/4.4.15-tegra/kernel/drivers/usb/serial
	sudo mkdir -p /lib/modules/4.4.15-tegra/kernel/drivers/usb/class
	sudo cp cdc-acm.ko /lib/modules/4.4.15-tegra/kernel/drivers/usb/class
fi

cp ~/2017VisionCode/.vimrc ~/2017VisionCode/.gvimrc ~
sudo cp ~/2017VisionCode/kjaget.vim /usr/share/vim/vim74/colors

git config --global user.email "progammers@team900.org"
git config --global user.name "Team900 Jetson TX1"
