"""Open the browser game locally: python serve.py  ->  http://localhost:8000"""
import http.server
import os
import socketserver
import webbrowser

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))
PORT = int(os.environ.get("PORT", 8000))
with socketserver.TCPServer(("127.0.0.1", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    url = f"http://localhost:{PORT}"
    print(f"Connect Zero running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()
