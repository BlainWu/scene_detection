import csv
import requests
def update_csv():
    rank_url = 'https://competitions.codalab.org/competitions/28113/results/46181/data'
    with requests.Session() as s:
        download = s.get(rank_url)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
    data =  my_list[1:] # delete the top line

    usr_dic = []
    for line in data:
        tmp = {'Name': line[1], 'Top-1': float(line[2].split(' ')[0]), 'Time': float(line[3].split(' ')[0])}
        usr_dic.append(tmp)
    #return a list of dict
    return usr_dic

def calculate_score(usr_dic):
    for line in usr_dic:

        if line['Name'] == 'stvea':
            line['Time'] = 5.0
            line['Top-1'] =95.5

        line['Score'] = ( 2** ( line['Top-1'] - 96.0 ) )/ line['Time']
    return usr_dic

def show_rank():
    usr_dic = update_csv()
    usr_dic = calculate_score(usr_dic)
    usr_dic.sort(key=lambda x: float(x['Score']),reverse=True)
    for index,line in enumerate(usr_dic):
        print('Rank:%2d Name:%15s  Top1:%.2f  Time:%.2f  Score:%.4f' % \
              (index+1,line['Name'],line['Top-1'],line['Time'],line['Score']))

if __name__ == '__main__':
    show_rank()
