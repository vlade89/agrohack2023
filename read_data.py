import os
from bs4 import BeautifulSoup


def parse_resume(soup):
    try:
        name = soup.find(attrs={"class":"resume-block__title-text"}).text
    except:
        name = ""

    try:
        gender = soup.find("span",attrs={"data-qa":"resume-personal-gender"}).text
    except:
        gender = ""

    try:
        age = soup.find("span",attrs={"data-qa":"resume-personal-age"}).text.replace("\xa0", " ")
    except:
        age = ""

    try:
        salary = soup.find(attrs={"class":"resume-block__salary"}).text.replace("\u2009","").replace("\xa0"," ")
    except:
        salary = ""

    try:
        experience = [exp.text.replace("[\t\r\n]", " ") for exp in soup.find_all("div",attrs={"data-qa":"resume-block-experience-description"})]
    except:
        experience = ""

    try:
        education = [ed.text for ed in soup.find_all("div",attrs={"data-qa":"resume-block-education-name"})]
    except:
        education = []

    try:
        tags = [tag.text for tag in soup.find(attrs={"class":"bloko-tag-list"}).find_all("span",attrs={"class":"bloko-tag__section bloko-tag__section_text"})]
    except:
        tags = []
    resume = {
        "name":name,
        "gender": gender,
        "age": age,
        "salary":salary,
        "experience": experience,
        "education": education,
        "tags":tags,
    }
    return resume


def read_resume(input_file):
    with open('./data/' + input_file, 'r') as fobj:
        resume = fobj.read()

    soup = BeautifulSoup(resume, "html.parser")

    return soup


if __name__ == "__main__":
    for file in os.listdir('./data'):
        my_soup = read_resume(file)
        resume = parse_resume(my_soup)

        print(my_soup)
