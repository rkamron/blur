import socket
import blur_faces1
from blur_faces1 import main_blurfaces1
from blur_plates import main_blurplates


def main():
	host = "0.0.0.0"
	port = 12345
	
	main_blurfaces1()
	main_blurplates()
	
	# Ben put in your code!!!!
	
	# Add in while loop later for socket transfer until somebody presses stop
	
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((host, port))
	server.listen(1)

	print(f"Server listening on port {port}...")

	conn, addr = server.accept()
	print(f"connected by {addr}")

	message = "lot:x,space:y,lat:j,long:k,occ.:0,res.:1"
	conn.sendall(message.encode())
	
	conn.close()
	server.close()

main()