import scrapy
from bs4 import BeautifulSoup


class QuotesSpider(scrapy.Spider):
    name = "narutospyder"
    allowed_domains = ['naruto.fandom.com']
    start_urls = ["https://naruto.fandom.com/wiki/Category:Jutsu"]

    # def start_requests(self):
    #     urls = [
    #         "https://quotes.toscrape.com/page/1/",
    #         "https://quotes.toscrape.com/page/2/",
    #     ]
    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # page = response.url.split("/")[-2]
        # filename = f"quotes-{page}.html"
        # Path(filename).write_bytes(response.body)
        # self.log(f"Saved file {filename}")

        div = response.css('div.category-page__members')  # Replace 'your-div-class' with the actual class or id

        # Extract all anchor tags within that div
        anchors = div.css('a')

        # Loop through the anchor tags and extract href and text
        for anchor in anchors:
            # yield {
            #     'text': anchor.css('::text').get().strip(),  # Get the text inside the anchor tag
            #     'href': anchor.css('::attr(href)').get()  # Get the href attribute
            # }

            href = anchor.css('::attr(href)').get()

            data_dict = scrapy.Request("https://naruto.fandom.com" + href, callback=self.jutsu_parser)

            yield data_dict

        ### Move to the next page
        for next_page in response.css('a.category-page__pagination-next wds-button wds-is-secondary '):
            yield response.follow(next_page, self.parse())

    def jutsu_parser(self, response):

        title = response.css('.mw-page-title-main::text').get().strip()

        # Extract all paragraphs and text within the content
        # content = response.css('div.mw-parser-output').getall()

        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')

        jutsu_type = ""
        jutsu_nature = ""
        jutsu_class = ""
        jutsu_range = ""

        my_dict = {}

        if soup.find('aside'):
            aside = soup.find('aside')

            for cell in aside.find_all('div', {'class': 'pi-data'}):
                if cell.find_all('h3'):
                    cell_name_1 = cell.find('h3').text.strip()
                    if cell_name_1 == "Classification":
                        jutsu_type = cell.find('div').text.strip()

                        my_dict[cell_name_1] = jutsu_type

                    cell_name_2 = cell.find('h3').text.strip()
                    if cell_name_2 == "Nature":
                        jutsu_nature = cell.find('div').text.strip()

                        my_dict[cell_name_2] = jutsu_nature

                    cell_name_3 = cell.find('h3').text.strip()
                    if cell_name_3 == "Class":
                        jutsu_class = cell.find('div').text.strip()

                        my_dict[cell_name_3] = jutsu_class

                    cell_name_4 = cell.find('h3').text.strip()
                    if cell_name_4 == "Range":
                        jutsu_range = cell.find('div').text.strip()

                        my_dict[cell_name_4] = jutsu_range

                    arr = ['Classification', 'Nature', 'Class', 'Range']

                    for ele in arr:
                        if ele in my_dict.keys():
                            pass
                        else:
                            my_dict[ele] = ""

        soup.find('aside').decompose()

        jutsu_description = soup.text.strip()

        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        my_dict['jutsu_description'] = jutsu_description

        return my_dict
