#include <stdio.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <errno.h>
#include <iostream>


class SerialPort {


    private:
        int ReadBufferSize;
        int timeout;
        char terminationChar;
        bool termination;
        std::string id;
        int baudRate;
        int fd;
        struct termios tty;

    public:

    SerialPort(int baudRate, std::string id)
    {
        Init(baudRate, id);
    }

    void Init(int baudRate, std::string id) {


        int USB = open(id.c_str(), O_RDWR| O_NOCTTY);
        if (USB < 0)
        {
            std::cerr << "Could not open " << id.c_str() << " as a TTY:";
            perror("");
        }
        
        memset(&tty, 0, sizeof(tty));
        this->baudRate = baudRate;
        this->id = id;
        

        cfsetospeed(&tty, (speed_t)baudRate);
        cfsetispeed(&tty, (speed_t)baudRate);

        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;

        //tty.c_cflag = CS8|CREAD|CLOCAL;

        //tty.c_cflag &= ~CRTSCTS;
        tty.c_cc[VMIN] = 1;
        tty.c_cc[VTIME] = 10;

        tty.c_cflag |= CREAD | CLOCAL;

        cfmakeraw(&tty);
        tcflush(this->fd, TCIOFLUSH);
        if(tcsetattr(USB,TCSANOW,&tty) != 0) std::cout << "Failed to initialize serial." << std::endl;

        this->fd = USB;
    }

    void SetReadBufferSize(int size) {
        this->ReadBufferSize = size;
    }

    void SetTimeout(int timeout) {
        this->timeout = timeout;
        tty.c_cc[VTIME] = timeout*10;
        cfmakeraw(&tty);
        if(tcsetattr(this->fd,TCSANOW,&tty) != 0) std::cout << "Failed to initialize serial in SetTimeout." << std::endl;;
    }


    void EnableTermination(char c) {
        this->termination = true;
        this->terminationChar = c;
    }


    void Flush() {
        tcflush(this->fd, TCOFLUSH);
    }

    void Write(char *data, int length) {
        int n_written = 0, spot = 0;
        do {

            n_written = write( this->fd, &data[spot], length );
            if (n_written > 0)
                spot += n_written;
        } while (data[spot-1] != terminationChar); 
    }

    int GetBytesReceived() {
        int bytes_avail;
        ioctl(this->fd, FIONREAD, &bytes_avail);
        return bytes_avail;
    }

    int Read(char *data, int size) {
        int n = 0, loc = 0;
        char buf = '\0';
        memset(data, '\0', size);

        do {
            n = read(this->fd, &buf, 1);
            sprintf( &data[loc], "%c", buf );
            loc += n;
        } while( buf != terminationChar && loc < size);

        if (n < 0) {
            std::cout << "Error reading: " << strerror(errno) << std::endl;
        }
        else if (n == 0) {
            std::cout << "Read nothing!" << std::endl;
        }        
        else {
            //std::cout << "Response: " << data  << std::endl;
        }
        return loc;
    }

    void WaitForData()
    {
	fd_set readfds;
	struct timeval tv;
	FD_ZERO(&readfds);
	FD_SET(this->fd, &readfds);
	tv.tv_sec = 0;
	tv.tv_usec = 100000;
	select(this->fd + 1, &readfds, NULL, NULL, &tv);
    }

    void Reset() {
        tcflush(this->fd, TCIOFLUSH);
    }

    void Close() {
        close(this->fd);
    }

};
