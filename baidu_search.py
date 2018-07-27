import requests
from lxml import etree, html


headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36',
    'Cookie': 'gr_user_id=1f9ea7ea-462a-4a6f-9d55-156631fc6d45; bid=vPYpmmD30-k; ll="118282"; ue="codin; __utmz=30149280.1499577720.27.14.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/doulist/240962/; __utmv=30149280.3049; _vwo_uuid_v2=F04099A9dd; viewed="27607246_26356432"; ap=1; ps=y; push_noty_num=0; push_doumail_num=0; dbcl2="30496987:gZxPfTZW4y0"; ck=13ey; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1515153574%2C%22https%3A%2F%2Fbook.douban.com%2Fmine%22%5D; __utma=30149280.833870293.1473539740.1514800523.1515153574.50; __utmc=30149280; _pk_id.100001.8cb4=255d8377ad92c57e.1473520329.20.1515153606.1514628010.'
}


def clear_content(content_list):
    res = ''
    for x in content_list:
        x = x.strip('\t').strip('\n').strip()
        if x:
            res += x

    print(res)


def search(keyword):
    base_url = 'https://www.baidu.com/s?tn=baidurt&wd={0}'
    
    search_url = base_url.format(keyword)

    res = requests.get(search_url, headers=headers)

    # selector = etree.HTML(res.text)
    selector = html.fromstring(res.text)

    contents = selector.xpath('//div[@class="content"]/table')

    temp = contents[1]

    real_time = temp.xpath('.//div[@class="realtime"]/text()')
    title = temp.xpath('.//h3[@class="t"]/a/text()')
    url = temp.xpath('.//h3[@class="t"]/a/@href')

    content_xpath = './/font[@size]'
    
    content = temp.xpath(content_xpath)[0]

    div_remove = temp.xpath('.//font[@size="-1"]/div')[0]
    a_remove = temp.xpath('.//font[@size="-1"]/a')[0]

    div_remove.drop_tree()
    a_remove.drop_tree()

    res = content.xpath('.//text()')
    clear_content(res)


if __name__ == '__main__':
    search('xhw')
