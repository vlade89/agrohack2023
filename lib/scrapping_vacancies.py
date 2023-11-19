import pickle
import time

import fake_useragent
import pandas as pd
import requests
from bs4 import BeautifulSoup

PATH_TO_DATA = '../data'


def get_links():
    ua = fake_useragent.UserAgent()
    html = requests.get(
        url=f"https://hh.ru/search/vacancy?L_save_area=true&text=&excluded_text=&professional_role=7&professional_role=19&professional_role=43&professional_role=63&professional_role=111&professional_role=49&industry=29&area=113&salary=&currency_code=RUR&experience=doesNotMatter&order_by=relevance&search_period=0&items_on_page=20",
        headers={"user-agent": ua.random}
    )
    if html.status_code != 200:
        return
    soup = BeautifulSoup(html.content, "lxml")

    try:
        page_count = int(
            soup.find("div", attrs={"class": "pager"})
            .find_all("span", recursive=False)[-1]
            .find("a").find("span").text)
        print(page_count)
    except:
        return
    for page in range(page_count):
        print(page)
        try:
            html = requests.get(
                url=f"https://hh.ru/search/vacancy?L_save_area=true&text=&excluded_text=&professional_role=7&professional_role=19&professional_role=43&professional_role=63&professional_role=111&professional_role=49&industry=29&area=113&salary=&currency_code=RUR&experience=doesNotMatter&order_by=relevance&search_period=0&items_on_page=20&page={page}",
                headers={"user-agent": ua.random}
            )
            if html.status_code == 200:
                soup = BeautifulSoup(html.content, "lxml")
                for a in soup.find_all("a", attrs={
                    # "class":"serp-item__title",
                    "data-qa": "serp-item__title"
                }):
                    link = f'{a.attrs["href"].split("?")[0]}'
                    print(link)
                    yield link
        except Exception as e:
            print(f"{e}")
        time.sleep(1)


def get_vacancy(link):
    ua = fake_useragent.UserAgent()
    data = requests.get(
        url=link,
        headers={"user-agent": ua.random}
    )
    if data.status_code != 200:
        return
    soup = BeautifulSoup(data.content, "lxml")

    try:
        name = soup.find(attrs={"class": "vacancy-title"}).text.replace("\xa0", " ")
    except:
        name = ""

    try:
        company = soup.find(attrs={"class": "g-user-content"}).text
    except:
        company = ""

    try:
        description = [desc.text for desc in soup.find_all("p", attrs={"class": "vacancy-description-list-item"})]
    except:
        description = ''

    try:
        salary = (
            soup.find(attrs={"class": "bloko-header-section-2 bloko-header-section-2_lite"})
            .text.replace("\u2009", "").replace("\xa0", " ")
        )
    except:
        salary = ""
    try:
        tags = [tag.text for tag in soup.find(attrs={"class": "bloko-tag bloko-tag_inline"}).find_all("span", attrs={
            "class": "bloko-tag__section_text"})]
    except:
        tags = []
    vacancy = {
        "name": name,
        "company": company,
        "description": description,
        "salary": salary,
        "tags": tags,
    }
    return vacancy


resumes = []
for a in get_links():
    res = get_vacancy(a)
    resumes.append(res)
    time.sleep(1)

data = pd.DataFrame.from_records(resumes)
pattern = "(?=от|до \d+.*)"
data['name'] = data['name'].str.replace(pattern, ' ')

with open(PATH_TO_DATA + "agro_vacances.pickle", "wb") as f:
    pickle.dump(data, f)
