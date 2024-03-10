# Dev
apt install python3.11-venv
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Prod
gunicorn -w 1 -b 0.0.0.0 'app:app' --daemon
ps -ef | grep gunicorn
kill -9 <pid>
