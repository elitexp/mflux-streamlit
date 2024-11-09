import os
import sys
import subprocess


def main():
    """Entry point for the mflux-streamlit command"""
    # Get the directory where app.py is located
    dirname = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(dirname, 'app.py')

    # Run streamlit with the app.py file
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])


if __name__ == '__main__':
    main()
