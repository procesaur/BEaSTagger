from os import path
import requests
import zipfile
import io


def get_model(file, name, xpath=""):

    if xpath == "":
        xpath = path.join(path.dirname(__file__), "data/models/" + name)

    if 'https' in file or 'http' in file:
        response = requests.get(file)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(xpath)

    elif path.isfile(file):
        with zipfile.ZipFile(file, 'r') as z:
            z.extractall(xpath)

