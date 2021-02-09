from keras.models import load_model
import numpy as np

fashionClasses = {
    0: "Bags Handbags",
    1: "Shoes Casual Shoes",
    2: "Shoes Heels",
    3: "Shoes Sports Shoes",
    4: "Topwear Kurtas",
    5: "Topwear Shirts",
    6: "Topwear Tops",
    7: "Topwear Tshirts",
    8: "Watches Watches",
}


logoClasses = {
    0: "Adidas",
    1: "Apple",
    2: "BMW",
    3: "Citroen",
    4: "Cocacola",
    5: "DHL",
    6: "Fedex",
    7: "Ferrari",
    8: "Ford",
    9: "Google",
    10: "HP",
    11: "Heineken",
    12: "Intel",
    13: "McDonalds",
    14: "Mini",
    15: "Nbc",
    16: "Nike",
    17: "Pepsi",
    18: "Porsche",
    19: "Puma",
    20: "RedBull",
    21: "Sprite",
    22: "Starbucks",
    23: "Texaco",
    24: "Unicef",
    25: "Vodafone",
    26: "Yahoo",
}


def FashionClassification(img):
    fashionModel = load_model("fashion.h5")
    img.resize((128, 128, 3), refcheck=False)
    image = np.expand_dims(img, axis=0)
    pred = fashionModel.predict(image)
    return fashionClasses[np.argmax(pred)]


def LogoClassification(img):
    logoModel = load_model("logoModel.h5")
    img.resize((240, 240, 3))
    image = np.expand_dims(img, axis=0)
    pred = logoModel.predict(image)
    return logoClasses[np.argmax(pred)]
