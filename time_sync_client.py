# This runs on the driverstation
# This is a CLIENT
import zmq
from datetime import datetime 
import time 
import subprocess
context = zmq.Context()
port = "5803"
ip = "localhost" #Jetson IP
socket = context.socket(zmq.REQ)
socket.connect("tcp://%s:%s" % (ip,port))

while True:
	time_string = datetime.now().isoformat()
	socket.send(time_string)
	message = socket.recv()
	time.sleep(0.5)
