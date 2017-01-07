#include <string>
#include "zca.hpp"

int main(int argc, char **argv)
{
	ZCA zca(argv[1], 0);
	zca.Print();
	//zca.Resize(atoi(argv[2]));
	//zca.WriteCompressed(argv[3]);
	//ZCA zca2(argv[3], 0);
	//zca2.Write(std::string(argv[3]) + ".xml");
}
