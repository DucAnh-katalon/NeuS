mkdir public_data
gdown --id 1zgD-uTLjO8hXcjLqelU444rwS9s9-Syg -O public_data/
unzip public_data/data_DTU.zip -d public_data/
pip install -r requirements.txt
python exp_runner.py --mode train --conf confs/womask.conf --case dtu_scan105