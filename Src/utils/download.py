import os, requests

def download_dataset(fname,url):
    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)
                    return True
        return False
    else:
        return True