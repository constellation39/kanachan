#!/usr/bin/env python3

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from _kanachan import Kanachan


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Properly handle arguments and call the super class's __init__ method
        super().__init__(*args, **kwargs)
        self.kanachan = Kanachan()


    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Hello, World!")


    def do_POST(self):
        # Read the POST data from the request
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        post_params = json.loads(post_data)  # Parse JSON data

        # Send a response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        

        response_data = self.kanachan.action(post_params)

        # Convert the dictionary to JSON format
        response_json = json.dumps(response_data)

        # Send the JSON response content
        self.wfile.write(response_json.encode('utf-8'))

# Define the server's address and port
server_address = ('', 8000)

# Create the HTTP server object
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

# Start the server
print("Server is running on http://127.0.0.1:8000/")
httpd.serve_forever()
