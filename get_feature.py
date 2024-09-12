import re
import csv
def match(output,l1):
    f = open(output)
    lines = f.readlines()
    la = len(lines)
    str1 = ''
    for i in range(0, l1-1):
        fid1 = lines[i].replace('\n', '').split('\t')[0]
        ll = lines[i].replace('\n', '').split('\t')[1]
        a = ll.split(',')
        lengths = len(a)
        for j in range(l1-1, la):
            fid2 = lines[j].replace('\n', '').split('\t')[0]
            lb = lines[j].replace('\n', '').split('\t')[1]
            b = lb.split(',')
            str1 = str1 + fid1 + '\t' + fid2
            for z in range(0, lengths):
                str1 = str1 + '\t' + a[z] + ' ' + b[z]
            str1 += '\n'
    f.close()
    return str1
def saveCSV(output,l1):
    csvFile = open('testContracts/test_pairs.csv', 'w')
    writer = csv.writer(csvFile)
    csvRow = []
    a=match(output,l1).split('\n')
    for line in a:
        if line!='\n' and line!='':
            csvRow = line.replace('\n', '').split('\t')
            writer.writerow(csvRow)
    csvFile.close()
def getfeature(a):
    b = ""
    elements = a.split('\'type\':')
    flag = False
    # print(elements)
    if len(elements)<2:
        return ''
    for e in elements:
        arr = e.split(',')
        if arr[0].strip() == "'FunctionCall'":
            num1 = arr[0].replace("'", "").replace("}", "").replace("{", "").replace(",", "").replace("'", "").replace(']',
                                                                                                             "").replace(
                    '[', "").replace("\n", "")
            b += num1
            flag = True
            continue
        elif arr[0] != "{" and arr[0]!="" and arr[0]!= None:
            if flag:
                element = e.split('name\':')
                element0 = element[len(element) - 1].split(',')[0]
                num1 = element0.replace("}", "").replace("{", "").replace(",", "").replace("'", "").replace(']',
                                                                                                                   "").replace(
                    '[', "").replace("\n", "")
                flag = False
                b += num1
            num1 = arr[0].replace("}", "").replace("{", "").replace(",", "").replace("'", "").replace(']',
                                                                                                      "").replace(
                '[', "").replace("\n", "")
            b += num1
    return b.replace('}','')
def main():
    inputpath = "testContracts/SRs.txt"
    outpath = "testContracts/Features.txt"
    f = open(inputpath)
    lines = f.readlines()
    feature = ""
    fun_id = ""
    str_id = 1
    l1=0
    locs=''
    for line in lines:
            if "ContractDefinition" not in line and "FunctionDefinition" not in line:
                if len(line.split('\t'))==3:
                    fun_id = line.split('\t')[0]
                    l1=str_id
                    str_id = 1
                    locs = line.split('\t')[1]
                    line = line.split('\t')[2]
                else:
                    locs = line.split('\t')[0]
                    line=line.split('\t')[1]
                if getfeature(line).strip() != '' and getfeature(line).strip() != None:
                    # feature = feature + str(fun_id) + '\t' + str(str_id) + '\t' + getfeature(line).strip() + '\n'
                    # str_id += 1
                    b = line.split(',')
                    all = ''
                    type=getfeature(line).strip()
                    a=type.split()
                    name=''
                    names=''
                    value=''
                    subdenomination=''
                    operator=''
                    memberName=''
                    for i in b:
                        res = i.split(':')[-1].replace('[', '').replace(' ', '').replace(']', '').replace('}','')\
                            .replace("'", '').replace('\t', '').replace('\n', '')
                        if res!='':
                            if 'type' in i:
                                continue
                            if 'names' in i:
                                names=names+' []'
                            elif 'name' in i:
                                name = name+' '+res
                            elif 'value' in i or 'number' in i:
                                value=value+' '+res
                            elif 'subdenomination' in i:
                                subdenomination=subdenomination+' '+res
                            elif 'operator' in i:
                                operator=operator+' '+res
                            elif 'memberName' in i:
                                memberName=memberName+' '+res
                            else :
                                if 'column' not in i and 'line' not in i:
                                     all = all+' '+res
                    for j in a:
                        all = all.replace(j, '', 1)
                        name = name.replace(j, '', 1)
                    feature = feature+str(fun_id)+'_'+str(str_id)+'_'+locs+'\t'+type+','+name+','+names+','+value+\
                              ','+subdenomination+','+operator+','+memberName+','+all.replace(',','').strip(',')+'\n'
                    str_id+=1
    f.close()
    ft = open(outpath, 'w')
    ft.write(feature)
    ft.close()
    saveCSV(outpath,l1)
main()
