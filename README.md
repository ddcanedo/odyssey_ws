# Machine Learning Web Service for ODYSSEY

## Table of Contents

-  [Installation](#installation)

## Installation

**NOTE**: *The installation assumes an Ubuntu 20.04LTS 64-bit environment.*

1. Clone the repository	

2. Install the dependencies
    ```bash
    sudo apt update
    sudo apt install python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv
    pip install -r requirements.txt
    ```
3. Create a virtual environment

    ```bash
    python3 -m venv ws
    source ws/bin/activate
    ```
4. Configure gunicorn and check if everything is working

    ```bash
    gunicorn --bind 0.0.0.0:5000 wsgi:app
    ```
    If it is working properly, leave the virtual environment:
    ```bash
    deactivate
    ```
5. Create the systemd service unit file to automatically start gunircorn on boot
    ```bash
    sudo nano /etc/systemd/system/ml.service
    ```
    
    Add the following and change the paths to accordingly:
    ```bash
    [Unit]
    Description=Gunicorn instance to serve ml
    After=network.target

    [Service]
    User=daniel
    Group=www-data
    WorkingDirectory=/home/daniel/Desktop/gunicornws
    Environment="PATH=/home/daniel/Desktop/gunicornws/ws/bin"
    ExecStart=/home/daniel/Desktop/gunicornws/ws/bin/gunicorn --workers 3 --timeout 0 --bind unix:ml.sock -m 007 wsgi:app

    [Install]
    WantedBy=multi-user.target
    ```
    
    Start the service:
    ```bash
    sudo systemctl start ml
    sudo systemctl enable ml
    sudo systemctl status ml
    ```
    
    If any changes are made to the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl restart ml
    ```
    
6. Configure Nginx

    ```bash
    sudo nano /etc/nginx/sites-available/ml
    ```
    
    Add the following and change the paths to accordingly:
    ```bash
server {
    listen 8080;
    location / {
        include proxy_params;
        proxy_pass http://unix:/home/daniel/Desktop/gunicornws/ml.sock;
    }
}
    ```
    
    Enable the configuration:
    ```bash
    sudo ln -s /etc/nginx/sites-available/ml /etc/nginx/sites-enabled
    sudo nginx -t
    sudo systemctl restart nginx
    sudo ufw allow 'Nginx Full'
    ```
    
    Give permissions to the socket file if needed (change the path):
    ```bash
    sudo chmod 755 /home/daniel
    ```
    
7. GeoNode
    If changes are made to the GeoNode, update it:
    ```bash
    docker-compose -f docker-compose.yml build --no-cache
    docker-compose -f docker-compose.yml up -d
    ```
