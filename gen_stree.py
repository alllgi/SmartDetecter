import os


def readFiles(inputDir,outputflie):
    script ='# !/bin/bash \n'
    for _,_, files in os.walk(inputDir):
        # print(files)
        for file in files:
            script = script + 'python3 -m solidity_parser parse '+inputDir+file+'\n'
            print(script)
    f = open(outputflie,'w')
    f.write(script)
    f.close()
def main():
    inputDir="./dataset/"
    outputflie="./gen_ast.sh"
    readFiles(inputDir,outputflie)
main()