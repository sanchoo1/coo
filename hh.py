import socket
import time

# Simulated IFM O2D camera values
def generate_camera_values(model_class, obj_cnt, x, y, z, rotation):
    while True:
        values = "0000startM"+model_class+"#"+obj_cnt+"O"+x+"#"+y+"#"+z+"#"+rotation+"#stop"
        yield values
        time.sleep(1)  # Simulating sending values every 1 second


# TCP server settings
HOST = '127.0.0.1'  # Change to the desired host IP
PORT = 50010       # Change to the desired port number

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        print(f"Listening on {HOST}:{PORT}")
        client_socket, client_addr = server_socket.accept()
        print(f"Accepted connection from {client_addr}")

        value_generator = generate_camera_values("3", "1", "-9", "1", "0", "-80")
        
        try:
            while True:
                camera_value = next(value_generator)
                client_socket.sendall(camera_value.encode())
        except KeyboardInterrupt:
            print("Server stopped by user.")
        finally:
            client_socket.close()

if __name__ == "__main__":
    main()
    
    
