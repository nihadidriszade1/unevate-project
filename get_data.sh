#!/usr/bin/env bash
# Analizlər üçün açıq datasetləri yükləyir
set -e
mkdir -p data
curl -L -o data/online_shoppers.zip "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
unzip -o data/online_shoppers.zip -d data
curl -L -o data/hillstrom.csv "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
echo "Hazırdır: data/online_shoppers_intention.csv və data/hillstrom.csv"
