set -euo pipefail

APP_DIR=/opt/fire-detector
SERVICE=fire-detector

apt-get update
apt-get install -y --no-install-recommends nginx rsync python3-venv ca-certificates

mkdir -p "$APP_DIR"
rsync -a --delete --exclude=".git" --exclude=".venv" ./ "$APP_DIR/"
chown -R www-data:www-data "$APP_DIR"

if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
  python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip wheel setuptools
if [ -f "$APP_DIR/requirements.txt" ]; then
  "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
fi

cat >/etc/systemd/system/${SERVICE}.service <<'EOF'
[Unit]
Description=Fire Detector (FastAPI + Uvicorn)
After=network.target

[Service]
Environment="PATH=/opt/fire-detector/.venv/bin"
Environment="FD_MAX_SIDE=1280" "FD_MAX_PIXELS=2000000" "FD_MAX_UPLOAD_BYTES=41943040" 
Environment="OMP_NUM_THREADS=1" "MKL_NUM_THREADS=1"
Environment="FD_UNET_TILE=1024" "FD_UNET_OVERLAP=128"
Environment="TORCH_NUM_THREADS=1" "TORCH_NUM_INTEROP_THREADS=1"
Environment="FD_UNET_TILE=768" "FD_UNET_OVERLAP=96" "UNET_TOPK=256"
Environment="FD_MAX_SIDE=1152" "FD_MAX_PIXELS=1500000"
User=www-data
Group=www-data
WorkingDirectory=/opt/fire-detector
ExecStart=/opt/fire-detector/.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-keep-alive 65
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/nginx/sites-available/${SERVICE} <<'EOF'
server {
    listen 80;
    server_name vm3496857.firstbyte.club 185.204.3.172 _;

    client_max_body_size 128M;

    error_page 413 = /413.json;
    location = /413.json {
        internal;
        default_type application/json;
        return 413 '{"detail":"Request entity too large"}';
    }

    error_page 502 504 = /upstream_error.json;
    location = /upstream_error.json {
        internal;
        default_type application/json;
        add_header X-Upstream-Status $upstream_status always;
        return 502 '{"detail":"Upstream error","upstream_status":"$upstream_status"}';
    }

    location /static/ {
        alias /opt/fire-detector/frontend/;
        access_log off;
        expires 30d;
    }

    location / {
        proxy_pass         http://127.0.0.1:8000;

        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        client_body_timeout 600s;
        proxy_connect_timeout 600s;
        send_timeout          600s;

        proxy_request_buffering off;
    }
}
EOF

ln -sfn /etc/nginx/sites-available/${SERVICE} /etc/nginx/sites-enabled/${SERVICE}

systemctl daemon-reload
systemctl enable --now ${SERVICE}

nginx -t
systemctl reload nginx