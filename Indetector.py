import os
import sys
import time

def run(command):
    path = './testContracts/'
    files = os.listdir(path)
    counts = 0
    names = []
    for file in files:
        if '.sol' in file:
            counts += 1
            names.append(file)
    if counts == 2:
        script = '# !/bin/bash\n'
        f1 = open('run.sh', 'w')
        script = script + 'rm -rf testContracts/SRs.txt\n'
        script = script + 'python3 -m solidity_parser parse ' + path + names[0] + '\n'
        script = script + 'python3 -m solidity_parser parse ' + path + names[1] + '\n'
        script = script + 'python get_feature.py\n'
        script = script + 'python lightgbm_smart.py ' + command + '\n'
        f1.write(script)
        f1.close()
        print("Star run.sh")
        os.system("./run.sh")
    else:
        print('The number of contracts tested must be two')
        exit(0)

if __name__ == "__main__":
    if not len(sys.argv) > 1 or sys.argv[1] not in ("--train", "--test"):
        print("\n- Missing subcommand.\n  Please choose --train or --test")
        sys.exit(1)
    if sys.argv[1] == "--train":
        run('--train')
    elif sys.argv[1] == "--test":
        run('--test')
