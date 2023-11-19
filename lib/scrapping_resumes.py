import pickle
import time
import logging
import fake_useragent
import pandas as pd
import requests
from bs4 import BeautifulSoup

PATH_TO_DATA = '../data'


def get_links():
    ua = fake_useragent.UserAgent()
    html = requests.get(
        url="https://hh.ru/search/resume?text=&logic=normal&pos=full_text&exp_period=all_time&exp_company_size=any&filter_exp_industry=29&filter_exp_period=last_year&area=113&relocation=living_or_relocation&age_from=&age_to=&professional_role=7&professional_role=19&professional_role=43&professional_role=63&professional_role=111&professional_role=49&gender=unknown&salary_from=&salary_to=&currency_code=RUR&order_by=relevance&search_period=0&items_on_page=20&no_magic=true",
        headers={"user-agent": ua.random}
    )
    if html.status_code != 200:
        return
    soup = BeautifulSoup(html.content, "lxml")

    try:
        page_count = int(
            soup.find("div", attrs={"class": "pager"}).find_all("span", recursive=False)[-1].find("a").find(
                "span").text)
    except:
        return
    for page in range(page_count):
        try:
            html = requests.get(
                url=f"https://hh.ru/search/resume?text=&logic=normal&pos=full_text&exp_period=all_time&exp_company_size=any&filter_exp_industry=29&filter_exp_period=last_year&area=113&relocation=living_or_relocation&age_from=&age_to=&professional_role=7&professional_role=19&professional_role=43&professional_role=63&professional_role=111&professional_role=49&gender=unknown&salary_from=&salary_to=&currency_code=RUR&order_by=relevance&search_period=0&items_on_page=20&no_magic=true&page={page}",
                headers={"user-agent": ua.random}
            )
            if html.status_code == 200:
                soup = BeautifulSoup(html.content, "lxml")
                for a in soup.find_all("a", attrs={"data-qa": "serp-item__title"}):
                    link = f'https://hh.ru{a.attrs["href"].split("?")[0]}'
                    yield link
        except Exception as e:
            print(f"{e}")
        time.sleep(1)


def get_resume(link):
    ua = fake_useragent.UserAgent()
    data = requests.get(
        url=link,
        headers={"user-agent": ua.random}
    )
    if data.status_code != 200:
        return
    soup = BeautifulSoup(data.content, "lxml")
    try:
        name = soup.find(attrs={"class": "resume-block__title-text"}).text
    except:
        name = ""
    try:
        gender = soup.find("span", attrs={"data-qa": "resume-personal-gender"}).text
    except:
        gender = ""
    try:
        age = soup.find("span", attrs={"data-qa": "resume-personal-age"}).text.replace("\xa0", " ")
    except:
        age = ""
    try:
        salary = soup.find(attrs={"class": "resume-block__salary"}).text.replace("\u2009", "").replace("\xa0", " ")
    except:
        salary = ""
    try:
        experience = [exp.text.replace("[\t\r\n]", " ") for exp in
                      soup.find_all("div", attrs={"data-qa": "resume-block-experience-description"})]
    except:
        experience = ""
    try:
        education = [ed.text for ed in soup.find_all("div", attrs={"data-qa": "resume-block-education-name"})]
    except:
        education = []
    try:
        tags = [tag.text for tag in soup.find(attrs={"class": "bloko-tag-list"}).find_all("span", attrs={
            "class": "bloko-tag__section bloko-tag__section_text"})]
    except:
        tags = []
    resume = {
        "name": name,
        "gender": gender,
        "age": age,
        "salary": salary,
        "experience": experience,
        "education": education,
        "tags": tags,
    }

    return resume


resumes = []
for a in get_links():
    res = get_resume(a)
    resumes.append(res)
    time.sleep(1)

data = pd.DataFrame.from_records(resumes)
logging.info('Total number of resumes: ', len(data))
with open(PATH_TO_DATA + "agro_resumes.pickle", "wb") as f:
    pickle.dump(data, f)
