import os
import subprocess
import webbrowser
import socket
import logging
import time
import sys
import atexit

# Global variables for process tracking
streamlit_process = None

# Constants
DEFAULT_PORT = 8501

def cleanup():
    """Cleanup function to terminate processes."""
    global streamlit_process
    
    if streamlit_process:
        logging.info("Terminating Streamlit...")
        try:
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
        except:
            streamlit_process.kill()

def find_free_port():
    """Find a free port, trying default first."""
    def is_port_free(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except OSError:
            return False

    # Try default port first
    if is_port_free(DEFAULT_PORT):
        return DEFAULT_PORT

    # Find random port if default is taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_streamlit(port):
    """Start Streamlit application."""
    logging.info(f"Starting Streamlit on port {port}")
    layout_path = os.path.join(os.path.dirname(__file__), 'layout.py')
    
    env = os.environ.copy()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env['PYTHONPATH'] = root_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    return subprocess.Popen([
        'streamlit', 'run',
        layout_path,
        '--server.port', str(port),
        '--server.address', '127.0.0.1',
        '--server.headless', 'true',
        '--server.maxUploadSize', '200'
    ], env=env)

# Only register cleanup at exit
atexit.register(cleanup)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    port = find_free_port()
    logging.info(f"Using port {port}")
    
    streamlit_process = start_streamlit(port)
    time.sleep(2)
    
    try:
        webbrowser.open(f'http://localhost:{port}')
        streamlit_process.wait()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        cleanup()