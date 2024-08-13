import os
from abc import ABC , abstractmethod
import requests
from bs4 import BeautifulSoup

class Scraper(ABC):
    """
    Abstract class for building and using various web scrapers depending on website
    """
    @abstractmethod
    def get_headlines(self) -> None:
        """
        function to request scrap headline titles from url given
        """


class MC_Scraper(Scraper):
    def __init__(self) -> None:
        """
        This class for retrieving data drom money control website using web scraper primary url used is https://www.moneycontrol.com/news/business/stocks/'
        """
        super().__init__()

        self.url = 'https://www.moneycontrol.com/news/business/stocks/'
        self.headlines = []

    def get_headlines(self) -> list[str]:
        """
        function for scraping news headline from money control website

        Args: None

        return : list[str]
        """
        try:
            req = requests.get(self.url)
            if not req.ok:
                raise Exception(f'Cannot reach web servers, error code : {req}')
        except Exception as e:
            raise e
        
        soup = BeautifulSoup(req.text,features='html.parser')
        soup = soup.find_all('h2')

        for i in soup:
            headline = i.find('a')
            try:
                headline = headline.get('title')
                self.headlines.append(headline)
            except:
                pass

        return self.headlines
