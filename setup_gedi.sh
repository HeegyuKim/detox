
git clone https://github.com/salesforce/GeDi.git
cd GeDi

'''Installing transformers v2.8'''

pip install transformers==2.8 datasets jsonlines
pip install -r hf_requirements.txt

'''Downloading GeDi topic model checkpoints'''
wget https://storage.googleapis.com/sfr-gedi-data/gedi_detoxifier.zip
unzip gedi_detoxifier.zip