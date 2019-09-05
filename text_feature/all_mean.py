sum = 0
with open('data/all_num.csv',encoding='utf-8') as f:
    for line in f:
        for i in line.strip().split(','):
            if int(i.strip()) >0 :
                sum = sum + int(i.strip())

    print(sum)

data = ""
with open('data/all_num.csv',encoding='utf-8') as f:
    for line in f:
        for i in line.strip().split(','):
            if int(i.strip()) >0 :
                j = int(i.strip())/sum
                j = int(i.strip()) + j * 7663136
            else:
                j = int(i.strip())
            data = data + str(j) +'\t'
        data = data + '\n'


with open('data/all_res.csv','w',encoding='utf-8') as f2:
    f2.write(data)

print('finished!!')