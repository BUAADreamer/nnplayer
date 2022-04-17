import os

import requests


class Crawler:
    def __init__(self, url, request_type='get'):
        self.url = url
        self.request_type = request_type

    def cache_file(self, url):
        return url.replace('/', '-')

    def download_from_url(self, url='', request_type='get', filename='timemachine', cache=True) -> str:
        if url == '':
            url = self.url
        if request_type == 'get':
            file_path = 'data\\' + filename + '.txt'
            # print(file_path)
            if cache:
                if os.path.exists(file_path):
                    return open(file_path).read()
                else:
                    res = requests.get(url).content.decode()
                    open(file_path, 'w').write(res)
                    return res
            else:
                res = requests.get(url).content.decode()
                return res
