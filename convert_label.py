import os
import csv

dir = 'cifar_3/test/'

def get_filename(dir):
	name = []
	for file in os.walk(dir):
		name.append(file)
	return name

file = open('test_label_2.txt','w+')
file_name = get_filename(dir)


#print(file_name[0][2][0].split('_')[1].split('.')[0])
for i in range(0,3000):
	file.write(file_name[0][2][i].split('_')[0]+'\n'+' '+file_name[0][2][i].split('_')[1].split('.')[0]+'\n')
	#file.write(file_name[0][2][i].split('_')[1].split('.')[0]+'\n')

file.close()
