# !/bin/bash
rm -rf testContracts/SRs.txt
python3 -m solidity_parser parse ./testContracts/batchTransfer.sol
python3 -m solidity_parser parse ./testContracts/transferProxy.sol
python get_feature.py
python lightgbm_smart.py --test
