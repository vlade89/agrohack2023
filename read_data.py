import os

from bs4 import BeautifulSoup


def resume_parser(input_file):
    with open('./data/' + input_file, 'r') as f:
        resume = f.read()
    soup = BeautifulSoup(resume, "lxml")
    print(soup)




if __name__ == "__main__":
    for file in os.listdir('./data'):
        resume_parser(file)
