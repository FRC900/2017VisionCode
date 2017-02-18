#!/bin/bash

username="Team900 Jetson TX1"
email="progammers@team900.org"

while [ $# -gt 0 ]
do
	case "$1" in
		-ab) username="Adithya Balaji";email="adithyabsk@gmail.com";;
		-aa) username="Alexander Allen";email="admin@altechcode.com";;
		-ag) username="Alon Greyber";email="alongreyber@gmail.com";;
		-kj) username="Kevin Jaget";email="kjaget@gmail.com";;
		-eb) username="Eric Blau";email="eblau1@gmail.com";;
		-h) echo >&2 \
			"usage: $0 [-ab, -aa, -ag, -kj, -eb, or -h]"
			exit 1;;
		*) break;; #terminate loop
	esac
	shift
done

git config --global user.name \""$username"\"
git config --global user.email \""$email"\"
