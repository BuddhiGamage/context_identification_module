import qi

class Connection:
    def __init__(self):
        self.session = qi.Session()

    def connect(self, ip, port):
        # Connect to the robot
        print("Connect to the robot...")
        try:
            self.session.connect("tcp://{0}:{1}".format(ip, port))
            print("Session Connected....!")
            return self.session
        except Exception as e:
            print("Could not connect to Pepper:", e)
            exit(1)
        # finally:
        #     # Close the connection
        #     self.session.close()