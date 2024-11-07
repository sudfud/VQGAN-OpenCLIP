import wget

def download_img(img_url: str) -> str | None:
    try:
        return wget.download(img_url, out = "input.jpg")
    except:
        return None