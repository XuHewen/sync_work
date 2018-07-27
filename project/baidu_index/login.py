from __future__ import absolute_import, print_function, unicode_literals

import time
import os
import json
import sys

import yaml
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from utils.logger import logger
from PIL import Image
from io import BytesIO
import pytesseract
from collections import defaultdict
from selenium.webdriver.chrome.options import Options


def login(baidu_info):

    url = 'https://index.baidu.com/'
    login_flag = False
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=2560,1600")

    try:
        browser = webdriver.Chrome(chrome_options=chrome_options)
        # browser = webdriver.Chrome()
        browser.get(url)

        account = baidu_info['account']
        passwd = baidu_info['passwd']
        cookie_path = baidu_info['cookie_path']

        if not os.path.exists(cookie_path):
            os.makedirs(cookie_path)

        cookie_path = os.path.join(cookie_path, 'baidu_index.json')

        if os.path.exists(cookie_path):
            with open(cookie_path, 'r') as f:
                cookies = json.loads(f.read())

            browser.delete_all_cookies()

            for cookie in cookies:
                browser.add_cookie(cookie)

            browser.get(url)

            element_user = WebDriverWait(browser, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//span[@class="username-text"]'))
            )

            if element_user.text == account:
                print('loging successful')
                login_flag = True

                cookies = browser.get_cookies()

                logger.log.info('saving cookies ... ')
                with open(cookie_path, 'w') as f:
                    json.dump(cookies, f)

                return browser

        if not login_flag:
            # 点击登录
            browser.delete_all_cookies()
            browser.get(url)

            element = WebDriverWait(browser, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//span[@class="username-text"]'))
            )
            element.click()

            element_user = WebDriverWait(browser, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//input[@name="userName"]'))
            )
            element_user.clear()
            element_user.send_keys(account)
            time.sleep(2.1213)

            element_passwd = browser.find_element_by_xpath(
                '//input[@name="password"]')
            element_passwd.clear()
            element_passwd.send_keys(passwd)
            time.sleep(5.12341321)

            element_submit = browser.find_element_by_xpath(
                '//input[@type="submit"]')
            element_submit.click()
            time.sleep(3.2342)

            element_user = WebDriverWait(browser, 20).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '//span[@class="username-text"]'))
            )

            if element_user.text == account:
                print('loging successful')
                login_flag = True

                cookies = browser.get_cookies()

                logger.log.info('saving cookies ... ')
                with open(cookie_path, 'w') as f:
                    json.dump(cookies, f)

                return browser

    except Exception as e:
        logger.log.exception('login error: %s' % e)
        browser.close()

    return None


def encode_keyword(keyword):
    keyword = keyword.encode('gbk')
    keyword = str(keyword).split('\\x')
    keyword = '%'.join(keyword)
    keyword = keyword[2:-1]

    return keyword


def binarizing(img, threshold):
    img = img.convert('L')

    pixdata = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


def vertical(img):
    pixdata = img.load()
    w, h = img.size
    ver_list = []
    cuts = []
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 0:
                black += 1
        ver_list.append(black)
    
    l, r = 0, 0
    flag = False

    for i, count in enumerate(ver_list):

        if count > 0 and not flag:
            l = i
            flag = True
        if (count == 0 and flag):
            r = i - 1
            flag = False
            cuts.append((l, r))

    return cuts


def horizon(img):
    pixdata = img.load()
    w, h = img.size
    ver_list = []
    cuts = []
    for x in range(h):
        black = 0
        for y in range(w):
            if pixdata[y,x] == 0:
                black += 1
        ver_list.append(black)
    
    l, r = 0, 0
    flag = False

    for i, count in enumerate(ver_list):

        if count > 0 and not flag:
            l = i
            flag = True
        if (count == 0 and flag):
            r = i - 1
            flag = False
            cuts.append((l, r))
    
    length = cuts[0][1] - cuts[0][0] + 1
    
    return length


def parse_value(browser, xpath_str, image, flag):

    res_value = [None, None]

    for i in range(2):
        xpath_temp = xpath_str.format(str(i+1))

        element_image = browser.find_element_by_xpath(xpath_temp)

        loc, size = element_image.location, element_image.size

        left = loc['x'] * 2
        top = loc['y'] * 2
        right = loc['x'] * 2 + size['width'] * 2 + 20
        bottom = loc['y'] * 2 + size['height'] * 2

        if flag:
            right = right - 60

        sub_image = image.crop((left, top, right, bottom))
        new_size = [x * 4 for x in sub_image.size]
        sub_image = sub_image.resize(new_size, Image.ANTIALIAS)
        sub_image = binarizing(sub_image, 200)
        # sub_image.show()
        pos = vertical(sub_image)
        
        res = []
        # sub_image.show()
        for x, y in pos:

            sub_sub_image = sub_image.crop((x, 0, y, new_size[1]))
            w, h = sub_sub_image.size
            # sub_sub_image.show()
            if horizon(sub_sub_image) > h / 3. or w > h / 6.:
                value = pytesseract.image_to_string(sub_sub_image, lang="eng",
                                                    config="--psm 10 -c tessedit_char_whitelist=-0123456789%")
                res.append(value)
        
        res_value[i] = ''.join(res)
    
    return res_value


def parse_screen(browser, xpath_str, image, image_path):

    element_image = browser.find_element_by_xpath(xpath_str)
    loc, size = element_image.location, element_image.size

    left = 30
    top = 120 - 40
    right = left + size['width'] * 2 - 10
    bottom = top + size['height'] * 2

    sub_image = image.crop((left, top, right, bottom))
    sub_image.save(image_path)


def search_index_single(keywords, baidu_info, browser=None):

    if not browser:
        browser = login(baidu_info)

    value = []

    keywords_encode = [encode_keyword(x) for x in keywords]

    screen_dir = baidu_info['screen_shot_path']
    if not os.path.exists(screen_dir):
        os.makedirs(screen_dir)

    for i, key in enumerate(keywords_encode):
        url = 'https://index.baidu.com/?tpl=trend&word={0}'.format(key)
        browser.get(url)
        time.sleep(1.11)

        try:
            browser.find_element_by_id('close-wj').click()
        except:
            pass

        screen_name = '{0}-screen.png'.format(keywords[i])
        screen_path = os.path.join(screen_dir, screen_name)

        png = browser.get_screenshot_as_png()
        image = Image.open(BytesIO(png))

        image_xpath1 = '//table[contains(@class, "mtable")]//tr/td[2]//span[@class="ftlwhf enc2imgVal"][{0}]'
        image_xpath2 = '//table[contains(@class, "mtable")]//tr/td[3]//span[@class="ftlwhf imgnums"][{0}]'
        image_xpath3 = '//table[contains(@class, "mtable")]//tr/td[4]//span[@class="ftlwhf imgnums"][{0}]'
        image_xpath4 = '//div[@id="trend"]'

        temp = []
        temp.append(keywords[i])
        value1 = parse_value(browser, image_xpath1, image, False)
        temp.extend(value1)
        value2 = parse_value(browser, image_xpath2, image, True)
        temp.extend(value2)
        value3 = parse_value(browser, image_xpath3, image, True)
        temp.extend(value3)

        value.append(temp)

        scroll_size = 400
        browser.execute_script('$(window).scrollTop(%d);' % scroll_size)

        png = browser.get_screenshot_as_png()
        image = Image.open(BytesIO(png))

        parse_screen(browser, image_xpath4, image, screen_path)
        
    return value, screen_path, browser


def search_index_mult(keywords, baidu_info, browser=None):

    if not browser:
        browser = login(baidu_info)

    value = []

    keywords_encode = [encode_keyword(x) for x in keywords]

    screen_dir = baidu_info['screen_shot_path']
    if not os.path.exists(screen_dir):
        os.makedirs(screen_dir)

    keys = '%2C'.join(keywords_encode)
    url = 'https://index.baidu.com/?tpl=trend&word={0}'.format(keys)
    browser.get(url)
    time.sleep(2.34242)

    try:
        browser.find_element_by_id('close-wj').click()
    except:
        pass

    png = browser.get_screenshot_as_png()
    image = Image.open(BytesIO(png))

    for i in range(len(keywords)):

        image_xpath1 = '//table[contains(@class, "mtable")]//tr[%s]/td[2]//span[@class="ftlwhf enc2imgVal"][{0}]' % str(i+2)
        image_xpath2 = '//table[contains(@class, "mtable")]//tr[%s]/td[3]//span[@class="ftlwhf imgnums"][{0}]' % str(i+2)
        image_xpath3 = '//table[contains(@class, "mtable")]//tr[%s]/td[4]//span[@class="ftlwhf imgnums"][{0}]' % str(i+2)
        

        temp = []
        temp.append(keywords[i])
        value1 = parse_value(browser, image_xpath1, image, False)
        temp.extend(value1)
        value2 = parse_value(browser, image_xpath2, image, True)
        temp.extend(value2)
        value3 = parse_value(browser, image_xpath3, image, True)
        temp.extend(value3)

        value.append(temp)

    image_xpath4 = '//div[@id="trend"]'
    screen_name = '{0}-screen.png'.format('-'.join(keywords))
    screen_path = os.path.join(screen_dir, screen_name)

    if len(keywords) == 3:
        n = 100
    else:
        n = 200

    scroll_size = 400 + n
    browser.execute_script('$(window).scrollTop(%d);' % scroll_size)

    png = browser.get_screenshot_as_png()
    image = Image.open(BytesIO(png))

    parse_screen(browser, image_xpath4, image, screen_path)

    return value, screen_path, browser



# # if __name__ == '__main__':
# #     login()
