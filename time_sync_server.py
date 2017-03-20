# This runs on the robot
# This is a SERVER 
import zmq
import time
import subprocess
context = zmq.Context()
port = "5803"

socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)
while True:
	message = socket.recv() #waits for time message
	socket.send("Thanks!")
	subprocess.call(["sudo", "time", message])
